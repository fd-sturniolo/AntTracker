"""Este archivo es un workaround por la estructura de módulos de AntLabeler y AntTracker.

AntLabeler debería usar labeler.classes.LabelingInfo, que tiene la
capacidad de guardar a partir de la estructura de clases de AntLabeler.

Cualquier otro uso que involucre leer los datos contenidos en un .tag
debería usar tracker.ant_labeler_info.LabelingInfo.

Cuidado, ambos archivos deben modificarse en conjunto para evitar incompatibilidades.

TODO: generar una solución que atienda a ambos casos, independientemente de cv2/pyav
"""
from dataclasses import dataclass, field

import itertools
import numpy as np
import ujson
from packaging.version import Version
from pathlib import Path
from typing import ClassVar, List, Union, TypedDict

from .common import Position, to_json, to_tuple
from .info import TracksInfo
from .track import Track

def groupSequence(lst: List[int]):
    # Visto en https://stackoverflow.com/a/2154437
    from operator import itemgetter
    ranges = []
    lst.sort()
    for k, g in itertools.groupby(enumerate(lst), lambda x: x[0] - x[1]):
        group = (map(itemgetter(1), g))
        group = list(map(int, group))
        ranges.append((group[0], group[-1]))
    return ranges

class UnlabeledFrame:
    def __init__(self, frame: int, contours):
        self.contours = [np.array(c) for c in contours]
        self.frame = frame

    def __repr__(self):
        return f"Frame: {self.frame}, {len(self.contours)} unlabeled contours"

    class Serial(TypedDict):
        frame: int
        contours: List[List[Position]]

    def encode(self) -> 'UnlabeledFrame.Serial':
        d = {
            "frame":    self.frame,
            "contours": [[
                to_tuple(point) for point in contour
            ] for contour in self.contours],
        }
        return d

    @staticmethod
    def decode(unlabeled_as_dict: 'UnlabeledFrame.Serial') -> 'UnlabeledFrame':
        return UnlabeledFrame(frame=unlabeled_as_dict['frame'], contours=unlabeled_as_dict['contours'])

# noinspection DuplicatedCode
@dataclass
class LabelingInfo(TracksInfo):
    unlabeledFrames: List[UnlabeledFrame] = field(init=False)
    labeler_version: Version = field(init=False)
    file_extension: ClassVar = '.tag'

    def __init__(self):
        raise AttributeError

    class Serial(TracksInfo.Serial):
        unlabeled_frames: List[UnlabeledFrame.Serial]
        labeler_version: str

    def encode(self) -> 'LabelingInfo.Serial':
        return {
            **super(LabelingInfo, self).encode(),
            'unlabeled_frames': [uf.encode() for uf in self.unlabeledFrames],
            'labeler_version':  str(self.labeler_version),
        }

    @classmethod
    def decode(cls, info: 'LabelingInfo.Serial'):
        labeler_version = Version(info.get('labeler_version', "1.0"))
        if labeler_version < Version("2.0"):
            raise ValueError(_version_error_msg(labeler_version))
        if labeler_version < Version("2.1"):
            info['tracks'] = _flip_contours_before_2_1(info['tracks'])

        self = super(LabelingInfo, cls).decode(info)
        self.labeler_version = labeler_version
        ufs = [UnlabeledFrame.decode(uf) for uf in info['unlabeled_frames']]
        self.unlabeledFrames = [uf for uf in ufs if uf.contours]
        return self

    def serialize(self, pretty=False) -> str:
        if pretty: return to_json(self.encode())
        return ujson.dumps(self.encode())

    def save(self, file: Union[Path, str], pretty=False):
        if not isinstance(file, Path):
            file = Path(file)
        if not self._is_extension_valid(file):
            raise ValueError(f'Wrong extension ({file.suffix}). Only {self.file_extension} files are valid.')
        with file.open('w') as f:
            f.write(self.serialize(pretty=pretty))

def _version_error_msg(current_version):
    return (f"Esta clase soporta sólo versión >=2.0 del protocolo. "
            f"Abra este archivo con una versión nueva de AntLabeler para actualizar. "
            f"Versión actual: {current_version}")

def _flip_contours_before_2_1(tracks: List[Track.Serial]):
    # had to do this to flip all x/y cause I did it wrong in AntLabeler
    for track in tracks:
        for blob in track['blobs'].values():
            blob['contour'] = [Position(p[1], p[0]) for p in blob['contour']]
    return tracks
