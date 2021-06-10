from dataclasses import dataclass, field
from enum import Enum

import cv2 as cv
import itertools
import numpy as np
import ujson
from packaging.version import Version
from pathlib import Path
from typing import ClassVar, List, NoReturn, Tuple, Dict, Union, Optional, NewType, TypedDict

from ..tracker.ant_labeler_info import groupSequence
from ..tracker.blob import Blob
from ..tracker.common import Position, to_json, to_tuple
from ..tracker.info import TracksInfo
from ..tracker.kellycolors import KellyColors
from ..tracker.parameters import SegmenterParameters, TrackerParameters
from ..tracker.track import Loaded, Track, TrackId
from ..version import __version__

CollectionVersion = Version(__version__)

Color = Tuple[int, int, int]
BinaryMask = NewType("BinaryMask", np.ndarray)
ColoredMask = NewType("ColoredMask", np.ndarray)
ColoredMaskWithUnlabel = NewType("ColoredMaskWithUnlabel", np.ndarray)
Vector = np.ndarray
FrameAndVelocity = Tuple[int, Vector]

class AreaInFrame:
    def __init__(self, frame: int, mask: BinaryMask):
        self.frame = frame
        self.area = (np.nonzero(mask))
        self.area = (self.area[0].tolist(), self.area[1].tolist())
        self.shape = mask.shape

    def encode(self) -> Dict:
        return {"frame": self.frame, "area": self.area}

    def getMask(self) -> BinaryMask:
        mask = np.zeros(self.shape, dtype='uint8')
        mask[self.area] = 1
        return BinaryMask(mask)

    @staticmethod
    def decode(area: Dict, shape) -> "AreaInFrame":
        areaInFrame = AreaInFrame(-1, BinaryMask(np.ndarray((0, 0))))
        areaInFrame.frame = area["frame"]
        areaInFrame.area = area["area"]
        areaInFrame.shape = shape
        return areaInFrame

class AreasByFrame:
    def __init__(self):
        self.areas_per_frame: List[AreaInFrame] = []

    def getArea(self, frame) -> Union[Tuple[int, AreaInFrame], Tuple[None, None]]:
        which = [(index, areaInFrame) for index, areaInFrame
                 in enumerate(self.areas_per_frame)
                 if areaInFrame.frame == frame]
        if len(which) == 1:
            return which[0][0], which[0][1]
        elif len(which) == 0:
            return None, None
        else:
            raise ValueError("More than one area in frame %d" % frame)

    def updateArea(self, frame: int, mask: BinaryMask):
        index, areaInFrame = self.getArea(frame)
        if not np.any(mask):
            if index is not None:
                self.areas_per_frame.pop(index)
        elif index is not None:
            self.areas_per_frame[index] = AreaInFrame(frame, mask)
        else:
            self.areas_per_frame.append(AreaInFrame(frame, mask))

    def encode(self):
        return [areaInFrame.encode() for areaInFrame in self.areas_per_frame]

    @staticmethod
    def decode(areas_as_list, shape) -> "AreasByFrame":
        areasByFrame = AreasByFrame()
        for area_as_dict in areas_as_list:
            areasByFrame.areas_per_frame.append(
                AreaInFrame.decode(area_as_dict, shape)
            )
        return areasByFrame

def epsilon(shape):
    size = shape[0] * shape[1]
    if size < 350000:
        return 0.01
    if size < 650000:
        return 0.03
    else:
        return 0.08

def get_contour(mask: BinaryMask):
    c, _ = cv.findContours(mask.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
    if not c: return []
    a = [cv.contourArea(cnt) for cnt in c]
    maxidx = a.index(max(a))
    contour = np.array(cv.approxPolyDP(c[maxidx], epsilon(mask.shape), True))
    contour = np.flip(contour.reshape((contour.shape[0], 2)), axis=1)
    return contour

def get_mask(contour, shape):
    mask = np.zeros(shape)
    if not np.any(contour): return mask
    pts = np.flip(contour, axis=1).reshape((-1, 1, 2))
    return cv.fillPoly(mask, [pts], 255).astype(bool)

class Ant:
    def __init__(self, _id: int):
        self.id = _id
        self.color = KellyColors.get(_id)
        # self.icon = ColorIcon(*self.color)
        self.loaded = False
        self.areasByFrame = AreasByFrame()

    def __repr__(self):
        ret = "Ant - Id: " + str(self.id) + "; Color: " + str(self.color)
        if self.loaded:
            ret += "; IsLoaded"
        return ret

    def updateArea(self, frame, mask):
        self.areasByFrame.updateArea(frame, mask)

    def getArea(self, frame):
        return self.areasByFrame.getArea(frame)[1]

    def isInFrame(self, frame):
        return self.areasByFrame.getArea(frame) != (None, None)

    def getMasksToUnlabel(self):
        areaInFrame: AreaInFrame
        frames_and_masks = [(areaInFrame.frame, areaInFrame.getMask())
                            for areaInFrame in self.areasByFrame.areas_per_frame]
        # print(str(frames_and_masks))
        return frames_and_masks

    def getMask(self, frame) -> Optional[BinaryMask]:
        areaInFrame: AreaInFrame
        _, areaInFrame = self.areasByFrame.getArea(frame)
        if not areaInFrame:
            return None
        else:
            return areaInFrame.getMask()

    def getInvolvedFrames(self) -> List[int]:
        areaInFrame: AreaInFrame
        return [areaInFrame.frame for areaInFrame in self.areasByFrame.areas_per_frame]

    def getLastFrame(self):
        return max(self.getInvolvedFrames(), default=0)

    def getGroupsOfFrames(self) -> List[Tuple[int, int]]:
        return groupSequence(self.getInvolvedFrames())

    def as_track(self):
        i = self.id
        blobs = dict()
        areas = self.areasByFrame.areas_per_frame
        for area_in_frame in areas:
            frame, mask = area_in_frame.frame, area_in_frame.getMask()
            blob = Blob(imshape=mask.shape, contour=get_contour(mask))
            blobs[frame] = blob
        # noinspection PyTypeChecker
        blobs = dict(sorted(blobs.items()))
        return Track(TrackId(i - 1), blobs, force_load_to=self.loaded)

    @staticmethod
    def from_track(track: Track, shape: Tuple[int, int]):
        self = Ant(track.id + 1)
        self.loaded = Loaded.to_bool(track.loaded)
        self.areasByFrame = AreasByFrame()
        for frame, blob in track.blobs.items():
            mask = get_mask(blob.contour, shape)
            self.areasByFrame.updateArea(frame, mask)
        return self

    def encode(self):
        return dict({
            "id":           self.id,
            "loaded":       self.loaded,
            "areasByFrame": self.areasByFrame.encode()
        })

    @staticmethod
    def decode(ant_as_dict, shape) -> "Ant":
        ant = Ant(-1)
        ant.id = ant_as_dict["id"]
        ant.loaded = ant_as_dict["loaded"]
        ant.areasByFrame = AreasByFrame.decode(ant_as_dict["areasByFrame"], shape)
        return ant

class UnlabeledFrame:
    def __init__(self, frame: Optional[int] = None, mask=None, _l=None, _i=None, _v=None, _f=None, contours=None):
        if frame is None:
            if _l is not None:
                self.frame = _f
                self.length = _l
                self.indices = _i
                self.values = _v
                return
            else:
                raise TypeError("Frame & Mask || Frame & contours || setters")
        elif mask is not None:
            contours, _ = cv.findContours(mask.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
            contours = [cv.approxPolyDP(c, epsilon(mask.shape), True) for c in contours]
            contours = [c.reshape(c.shape[0], 2) for c in contours]
            self.contours = contours
        elif contours is not None:
            self.contours = [np.array(c) for c in contours]
        else:
            raise TypeError("Frame & Mask || Frame & contours || setters")
        self.frame = frame
        #
        # packed_mask = np.packbits(mask,axis=None)
        # self.length = len(packed_mask)
        # self.indices = np.nonzero(packed_mask)
        # self.indices = (self.indices[0].tolist())
        # self.values = packed_mask[self.indices].tolist()

    def __repr__(self):
        return f"Frame: {self.frame}, {len(self.contours)} unlabeled contours"

    def getMask(self, shape):
        mask = cv.fillPoly(np.zeros(shape), self.contours, 255)
        return BinaryMask(mask.astype(bool))

    class Serial(TypedDict):
        frame: int
        contours: List[List[Position]]

    class OldSerial(TypedDict):
        frame: int
        length: int
        indices: List[int]
        values: List[int]

    def encode(self) -> 'UnlabeledFrame.Serial':
        d = {
            "frame":    self.frame,
            "contours": [[
                to_tuple(point) for point in contour
            ] for contour in self.contours],
        }
        return d

    @staticmethod
    def decode(unlabeled_as_dict: Union['UnlabeledFrame.OldSerial', 'UnlabeledFrame.Serial'], shape=None,
               size=None) -> "UnlabeledFrame":
        if 'values' in unlabeled_as_dict:
            # OldSerial
            def old_getMask(uf, _shape, _size) -> BinaryMask:
                """Get a binary mask with ones on segmented pixels"""
                packed_mask = np.zeros(uf.length, dtype='uint8')
                packed_mask[uf.indices] = uf.values
                mask = np.unpackbits(packed_mask, axis=None)[:_size].reshape(_shape)
                return BinaryMask(mask)

            u = UnlabeledFrame(
                _l=unlabeled_as_dict["length"],
                _i=unlabeled_as_dict["indices"],
                _v=unlabeled_as_dict["values"],
                _f=unlabeled_as_dict["frame"],
            )
            contours, _ = cv.findContours(old_getMask(u, shape, size).astype('uint8'), cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_TC89_L1)
            contours = [cv.approxPolyDP(c, epsilon(shape), True) for c in contours]
            contours = [c.reshape(c.shape[0], 2) for c in contours]
            u.contours = contours
        else:
            u = UnlabeledFrame(frame=unlabeled_as_dict['frame'], contours=unlabeled_as_dict['contours'])
        return u

def get_track(tracks: List[Track], ant_id):
    return [track for track in tracks if track.id == ant_id - 1][0]

class AntCollection:
    def __init__(self, anymask: Optional[np.ndarray] = None, video_length=None, info=None):
        self.ants: List[Ant] = []
        self.id_iter = itertools.count(start=1)
        self.videoSize = anymask.size if anymask is not None else 0
        self.videoShape = anymask.astype('uint8').shape if anymask is not None else (0, 0)
        if video_length is not None:
            self.videoLength = video_length
        self.getUnlabeledMask = self.__getUnlabeledMaskClosure(self.videoShape)
        self.info: LabelingInfo = info
        # ! versión inexistente mayor a todas las otras, para evitar migraciones
        # TODO: #9
        self._old_version = Version("2.2")
        self.version = CollectionVersion

    @staticmethod
    def __getUnlabeledMaskClosure(shape):
        def getMask(unl: UnlabeledFrame) -> BinaryMask:
            return unl.getMask(shape)

        return getMask

    def newAnt(self) -> Ant:
        _id = next(self.id_iter)
        ant = Ant(_id)
        self.ants.append(ant)
        track = Track(TrackId(_id - 1), {})
        self.info.tracks.append(track)
        return ant

    def getAnt(self, ant_id) -> Optional[Ant]:
        which = [ant for ant in self.ants if ant.id == ant_id]
        if len(which) == 1:
            return which[0]
        elif len(which) == 0:
            return None
        else:
            raise ValueError("More than one ant with id %d" % ant_id)

    def deleteAnt(self, ant_id):
        # Desetiquetar todas las áreas
        dead_ant = self.getAnt(ant_id)
        print("deleteAnt: dead_ant:", str(dead_ant))
        if dead_ant is None:
            raise ValueError("Trying to delete a nonexistent ant with id %d" % ant_id)
        else:
            for frame, mask in dead_ant.getMasksToUnlabel():
                print("deleteAnt: frame:", str(frame))
                self.updateUnlabeledFrame(frame, mask)
        self.ants.remove(dead_ant)
        dead_track = get_track(self.info.tracks, ant_id)
        self.info.tracks.remove(dead_track)

    def update_load(self, ant_id, loaded: bool):
        self.getAnt(ant_id).loaded = loaded
        get_track(self.info.tracks, ant_id)._Track__loaded = Loaded.parse(loaded)

    def getUnlabeledFrameGroups(self):
        unl = []
        for frame in self.info.unlabeledFrames:
            if len(frame.contours) > 0:
                unl.append(frame.frame)
        return groupSequence(unl), len(unl)

    def serialize(self) -> NoReturn:
        raise DeprecationWarning("Do not serialize as collection! Create a LabelingInfo instance instead")
        # return to_json({
        #     "ants":             [ant.encode() for ant in self.ants],
        #     "unlabeledFrames":  [uF.encode() for uF in self.unlabeledFrames],
        #     "videoSize":        self.videoSize,
        #     "videoShape":       self.videoShape,
        #     "version":          str(CollectionVersion)
        # })

    @staticmethod
    def deserialize(video_path, jsonstring=None, filename=None) -> "AntCollection":
        info = LabelingInfo.deserialize(jsonstring=jsonstring, filename=filename)
        antCollection = AntCollection.from_info(info)
        return antCollection

    def updateAreas(self, frame: int, colored_mask: ColoredMaskWithUnlabel):
        for ant in self.ants:
            contour = get_contour(colored_mask == ant.id)
            mask = get_mask(contour, self.videoShape)
            ant.updateArea(frame, mask)

            has_blob_in_frame = np.any(mask)
            track = get_track(self.info.tracks, ant.id)
            if has_blob_in_frame:
                track.blobs[frame] = Blob(imshape=mask.shape, contour=contour)
            elif track.at(frame) is not None:
                track.blobs.pop(frame)

        # Marcar áreas como etiquetadas
        index, unlabeled = self.getUnlabeled(frame)
        if index is not None:
            unlabeled_mask = self.getUnlabeledMask(unlabeled)

            # Quedan sólo las que no tienen etiqueta y que falten etiquetar
            unlabeled_mask = np.logical_and(colored_mask == -1, unlabeled_mask)

            if np.any(unlabeled_mask):
                self.overwriteUnlabeledFrame(frame, unlabeled_mask)
            else:
                self.deleteUnlabeledFrame(frame)

    def addUnlabeledFrame(self, frame: int, mask: BinaryMask):
        if np.any(mask):
            uf = UnlabeledFrame(frame, mask)
            self.info.unlabeledFrames.append(uf)

    def deleteUnlabeledFrame(self, frame: int):
        index, unlabeled = self.getUnlabeled(frame)
        if index is not None:
            self.info.unlabeledFrames.remove(unlabeled)

    def overwriteUnlabeledFrame(self, frame: int, mask: BinaryMask):
        index, unlabeled = self.getUnlabeled(frame)
        if index is not None:
            self.deleteUnlabeledFrame(frame)
        self.addUnlabeledFrame(frame, mask)

    def updateUnlabeledFrame(self, frame: int, new_mask: BinaryMask):
        index, unlabeled_packed = self.getUnlabeled(frame)
        if index is None:
            self.addUnlabeledFrame(frame, new_mask)
        else:
            unlabeled_mask = self.getUnlabeledMask(unlabeled_packed)
            unlabeled_mask = np.logical_or(unlabeled_mask, new_mask).astype('uint8')
            self.overwriteUnlabeledFrame(frame, unlabeled_mask)

    def getUnlabeled(self, frame) -> Union[Tuple[int, UnlabeledFrame], Tuple[None, None]]:
        """Returns the `frame`th packed frame of unlabeled regions and its index in the list"""
        which = [(index, unlabeledFrame) for index, unlabeledFrame
                 in enumerate(self.info.unlabeledFrames)
                 if unlabeledFrame.frame == frame]
        if len(which) == 1:
            return which[0][0], which[0][1]
        elif len(which) == 0:
            return None, None
        else:
            raise ValueError("More than one packed mask in frame %d" % frame)

    def getMask(self, frame) -> ColoredMaskWithUnlabel:
        mask = np.zeros(self.videoShape).astype('int16')
        area: AreaInFrame
        ant: Ant
        for (ant_id, area) in ((ant.id, ant.getArea(frame)) for ant in self.ants if ant.isInFrame(frame)):
            antmask = area.getMask().astype(bool)
            mask[antmask] = (antmask.astype('int16') * ant_id)[antmask]

        _, unlabeledFrame = self.getUnlabeled(frame)
        if unlabeledFrame is not None:
            ulmask = unlabeledFrame.getMask(self.videoShape)
            mask[ulmask] = (unlabeledFrame.getMask(self.videoShape).astype('int16') * (-1))[ulmask]

        return ColoredMaskWithUnlabel(mask)

    def cleanUnlabeledAndAntOverlaps(self, frame: int):
        index, unlabeledFrame = self.getUnlabeled(frame)
        if index is not None:
            unlmask = unlabeledFrame.getMask(self.videoShape).astype('bool')
            for ant in self.ants:
                if ant.isInFrame(frame):
                    antmask = ant.getMask(frame).astype('bool')
                    unlmask: BinaryMask = np.logical_and(unlmask, ~antmask)
            self.overwriteUnlabeledFrame(frame, unlmask)

    def cleanErrorsInFrame(self, frame, for_specific_ant: Ant = None):
        _, unlabeledFrame = self.getUnlabeled(frame)
        if unlabeledFrame is None:
            mask = np.zeros(self.videoShape).astype('int16')
        else:
            mask = unlabeledFrame.getMask(self.videoShape).astype('int16') * (-1)
        # print("cleaning frame ", frame)
        if for_specific_ant is not None:
            for ant in self.ants:
                if ant.isInFrame(frame):
                    mask = mask + ant.getMask(frame).astype('int16') * ant.id
            alreadyPainted: BinaryMask = mask != 0
            aboutToPaint = for_specific_ant.getMask(frame)
            overlap: BinaryMask = np.logical_and(alreadyPainted, aboutToPaint)
            if np.any(overlap):
                for_specific_ant.updateArea(frame, np.zeros(self.videoShape))
        else:
            for ant in self.ants:
                if ant.isInFrame(frame):
                    # print("- cleaning ant ", ant.id)
                    alreadyPainted = mask != 0
                    aboutToPaint = ant.getMask(frame)
                    overlap: BinaryMask = np.logical_and(alreadyPainted, aboutToPaint)
                    if np.any(overlap):
                        ant.updateArea(frame, np.zeros(self.videoShape))
                    else:
                        mask = mask + ant.getMask(frame).astype('int16') * ant.id

    def cleanErrors(self, number_of_frames, for_specific_ant: Ant = None, from_this_frame=0):
        for frame in range(from_this_frame, number_of_frames):
            self.cleanErrorsInFrame(frame, for_specific_ant)

    def labelFollowingFrames(self, current_frame, ant_id, tracking_radius=160, conflict_radius=60):
        def centroids_no_background(mask):
            _, _, _, cents = cv.connectedComponentsWithStats(mask.astype('uint8'))
            return cents[1:]

        def closest_two_nodes(node, nodes):
            nodes = np.asarray(nodes)
            dist_2 = np.sum((nodes - node) ** 2, axis=1)
            index = dist_2.argsort()
            return nodes[index[:2]], dist_2[index[:2]]

        # Ordenar todos los frames que quedan después del actual
        # Por las dudas, en teoría ya deberían estar ordenados
        unlabeledFutureFrames = sorted(
            [uframe for uframe in self.info.unlabeledFrames
             if uframe.frame > current_frame],
            key=lambda uframe: uframe.frame)
        # Estamos en el último frame/no quedan más frames sin etiquetar hacia adelante:
        if not unlabeledFutureFrames:
            return
        ant = self.getAnt(ant_id)
        # track = get_track(self.info.tracks, ant_id)
        # TODO: maybe make it so you can retag tagged regions (not that essential)
        last_frame = unlabeledFutureFrames[0].frame - 1
        last_mask = ant.getMask(current_frame)
        if last_mask is None:
            raise ValueError("El frame del que se quiere rellenar no tiene una hormiga ya etiquetada")
            # last_mask = np.zeros_like(self.getUnlabeledMask(unlabeledFutureFrames[0]),dtype='uint8')
        for uFrame in unlabeledFutureFrames:
            unlabel_mask = self.getUnlabeledMask(uFrame)
            frame = uFrame.frame
            print("Frame: ", frame)
            if frame != last_frame + 1:
                print("Hubo un salto, no hay chances de ver overlap")
                break
            colored_mask = self.getMask(frame)
            if np.any(colored_mask == ant_id):
                print("En este frame ya hay una hormiga etiquetada con ese id")
                break

            last_centroid = centroids_no_background(last_mask)
            if len(last_centroid) != 1:
                # FIXME: En realidad esto sí puede suceder,
                # si el usuario trata de rellenar en un frame donde ya pintó con ese id
                # Probablemente lo mejor sea largar QDialog de error avisando que
                # está intentando hacer algo indebido
                raise ValueError("En la máscara anterior debería haber un solo centroide")
            last_centroid = last_centroid[0]

            centroids = centroids_no_background(unlabel_mask)
            if len(centroids) == 0:
                print("Nos quedamos sin hormigas")
                break
            elif len(centroids) == 1:
                print("Hay una sola hormiga, es probablemente la que buscamos...")
                dist = np.sum((centroids[0] - last_centroid) ** 2, axis=0)
                if dist > tracking_radius:
                    print("Está muy lejos, probablemente sea una que recién aparece en otro lado")
                    print("(o bien la hormiga es muy rápida...)")
                    break
                else:
                    x, y = np.int(centroids[0][0]), np.int(centroids[0][1]) # noqa
                    if colored_mask[y, x] == -1:
                        print("Floodfill(centroids[0])")
                        upcasted_mask = colored_mask.astype('int32')
                        cv.floodFill(image=upcasted_mask,
                                     mask=None,
                                     seedPoint=(x, y),
                                     newVal=ant_id,
                                     loDiff=0,
                                     upDiff=0)
                        colored_mask = upcasted_mask.astype('int16').copy()
                        self.updateAreas(frame, colored_mask)
                    else:
                        print("El centroide del área anterior no cae en un área etiquetable")
                        break
            else:
                print("Más de una hormiga, busquemos las más cercanas")
                closest, dist = closest_two_nodes(last_centroid, centroids)
                if dist[1] < conflict_radius:
                    print("Dos hormigas muy cerca de la anterior, cortemos")
                    break
                elif dist[0] > tracking_radius:
                    print("Está muy lejos, probablemente la hormiga que seguíamos se fue de cámara")
                    break
                else:
                    x, y = np.int(closest[0][0]), np.int(closest[0][1]) # noqa
                    if colored_mask[y, x] == -1:
                        print("Floodfill(centroids[0])")
                        upcasted_mask = colored_mask.astype('int32')
                        cv.floodFill(image=upcasted_mask,
                                     mask=None,
                                     seedPoint=(x, y),
                                     newVal=ant_id,
                                     loDiff=0,
                                     upDiff=0)
                        colored_mask = upcasted_mask.astype('int16').copy()
                        self.updateAreas(frame, colored_mask)
                    else:
                        print("El centroide del área anterior no cae en un área etiquetable")
                        break

            # Unos en la parte recién filleada
            last_mask = (colored_mask == ant_id).astype('uint8')
            last_frame = frame
            # self.cleanErrorsInFrame(frame,ant)
        return

    def getLastLabeledFrame(self):
        return max((ant.getLastFrame() for ant in self.ants), default=0)

    def getLastFrame(self):
        lastAntFrame = self.getLastLabeledFrame()
        lastUnlabeledFrame = max((unlabeledFrame.frame for unlabeledFrame in self.info.unlabeledFrames), default=0)
        return max(lastAntFrame, lastUnlabeledFrame)

    def getLastId(self):
        if len(self.ants) == 0:
            return 0
        else:
            return max([ant.id for ant in self.ants])

    def ants_as_tracks(self):
        return [ant.as_track() for ant in self.ants]

    @staticmethod
    def from_info(info: 'LabelingInfo'):
        self = AntCollection(np.zeros(info.video_shape, dtype="uint8"), video_length=info.video_length, info=info)
        self.ants = [Ant.from_track(track, info.video_shape) for track in info.tracks]
        self.id_iter = itertools.count(start=self.getLastId() + 1)
        self.getUnlabeledMask = self.__getUnlabeledMaskClosure(self.videoShape)
        if info._labeler_version is not None:
            self._old_version = info._labeler_version
        else:
            self.version = info.version
        return self

class SerializableEnum(str, Enum):
    def _generate_next_value_(self, start, count, last_values):
        return self

def first(iterable, condition=lambda x: True):
    """
    Returns the first element that satisfies `condition`. \n
    Returns `None` if not found.
    """
    return next((x for x in iterable if condition(x)), None)

# noinspection DuplicatedCode
@dataclass
class LabelingInfo(TracksInfo):
    unlabeledFrames: List[UnlabeledFrame] = field(init=False)
    _labeler_version: Optional[Version] = field(init=False, default=None)
    file_extension: ClassVar[str] = '.tag'

    def __init__(self, video_path, ants: List[Ant], unlabeled_frames: List[UnlabeledFrame]):

        super(LabelingInfo, self).__init__(
            video_path=video_path,
            tracks=sorted([ant.as_track() for ant in ants], key=lambda t: t.id),
            segmenter_parameters=SegmenterParameters.mock(),
            tracker_parameters=TrackerParameters.mock(),
            version=CollectionVersion,
        )
        self.unlabeledFrames: List[UnlabeledFrame] = [uf for uf in unlabeled_frames if uf.contours]

    class Serial(TracksInfo.Serial):
        unlabeled_frames: List[UnlabeledFrame.Serial]
        labeler_version: Optional[str]
        version: Optional[str]

    def encode(self) -> 'LabelingInfo.Serial':
        return {  # noqa
            **super(LabelingInfo, self).encode(),
            'unlabeled_frames': [uf.encode() for uf in self.unlabeledFrames],
            'version':  str(self.version),
        }

    @classmethod
    def decode(cls, info: 'LabelingInfo.Serial'):
        #TODO: #9
        version = info.get('version')
        if not version:
            labeler_version = Version(info.get('labeler_version', "1.0"))
            if labeler_version < Version("2.1"):
                info['tracks'] = _flip_contours_before_2_1(info['tracks'])
        self = super(LabelingInfo, cls).decode(info)
        if version:
            self.version = Version(version)
        else:
            self._labeler_version = labeler_version
        size = self.video_shape[0] * self.video_shape[1]
        ufs = [UnlabeledFrame.decode(uf, self.video_shape, size) for uf in info['unlabeled_frames']]
        self.unlabeledFrames = [uf for uf in ufs if uf.contours]
        return self

    def serialize(self, pretty=False) -> str:
        if pretty: return to_json(self.encode())
        return ujson.dumps(self.encode())

    def save(self, file: Union[Path, str], pretty=False):
        if not isinstance(file, Path):  # noqa
            file = Path(file)
        if not self._is_extension_valid(file):
            raise ValueError(f'Wrong extension ({file.suffix}). Only {self.file_extension} files are valid.')
        with file.open('w') as f:
            f.write(self.serialize(pretty=pretty))

def _flip_contours_before_2_1(tracks: List[Track.Serial]):
    for track in tracks:
        for blob in track['blobs'].values():
            blob['contour'] = [Position(p[1], p[0]) for p in blob['contour']]
    return tracks
