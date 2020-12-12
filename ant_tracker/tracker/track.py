from dataclasses import dataclass
from enum import auto

import numpy as np
from memoized_property import memoized_property
from typing import Dict, List, NewType, Optional, Tuple, Type, TypeVar, TypedDict

from .blob import Blob
from .common import Color, ColorImage, FrameNumber, SerializableEnum, Side
from .kellycolors import KellyColors

TrackId = NewType('TrackId', int)

class Loaded(SerializableEnum):
    Yes = auto()
    No = auto()
    Undefined = auto()

    @staticmethod
    def parse(b: Optional[bool]): return Loaded.Undefined if b is None else Loaded.Yes if b else Loaded.No

    def to_bool(self): return None if self == Loaded.Undefined else True if self == Loaded.Yes else False

@dataclass
class Track:
    id: TrackId
    # Should be invariantly sorted, as frames are inserted sequentially and without gaps
    blobs: Dict[FrameNumber, Blob]

    # noinspection PyShadowingBuiltins
    def __init__(self, id: TrackId, blobs: Dict[FrameNumber, Blob], *, force_load_to: Optional[bool] = None):
        """
        Build a track from a set of `blobs`, with `id`. Currently encompasses labeled ants as well,
        using `force_load_to` to set the load state.

        TODO: split labeled ants and tracks into different classes.
        """
        self.id = id
        self.blobs = blobs
        self.__load_probability = None
        if force_load_to is not None:
            self.__load_probability = 1 * force_load_to
        self.__loaded = Loaded.parse(force_load_to)

    def __repr__(self):
        return f"Track(T{self.id}, " + (
            f"loaded={self.loaded})" if self.loaded != Loaded.Undefined else f"load_prob={self.load_probability}")

    @property
    def loaded(self) -> Loaded:
        """Returns either ``Loaded.Yes`` or ``Loaded.No`` if `self` is a labeled ant,
        or ``Loaded.Undefined`` if it's a track."""
        return self.__loaded

    def set_load_probability(self, prob):
        """Use a ``LeafDetector`` instance to get `prob`"""
        self.__load_probability = prob

    @property
    def load_probability(self):
        """Returns probability of track being loaded. If `self` is a labeled ant, use ``loaded`` instead."""
        return self.__load_probability

    @property
    def load_detected(self):
        return self.__load_probability is not None

    @property
    def load_prediction(self):
        """Returns a prediction based on load probability."""
        return self.__load_probability > 0.5

    @property
    def load_certainty(self):
        """Returns certainty of prediction given by `load_prediction`"""
        return abs(self.load_probability - 0.5) + 0.5

    @property
    def color(self):
        return KellyColors.get(self.id)

    def at(self, frame: FrameNumber):
        return self.blobs.get(frame, None)

    def path(self) -> np.ndarray:
        """Returns an array of shape (3, n_blobs), where dim0 is (x, y, frame_number)"""
        return np.array([[blob.center_xy.x, blob.center_xy.y, frame] for frame, blob in self.blobs.items()])

    def direction_of_travel(self, imshape: Tuple[int, int], percentage=0.1) -> Tuple[Side, Side]:
        first, last = self.first_blob().center_xy, self.last_blob().center_xy
        return Side.from_point(first, imshape, percentage), Side.from_point(last, imshape, percentage)

    @memoized_property
    def velocity_mean(self):
        if len(self.blobs) < 2:
            return np.array([0, 0])
        return (self.last_blob().center - self.first_blob().center) / (self.last_frame() - self.first_frame())

    @memoized_property
    def speed_mean(self):
        return self.speed_lowpass.mean()

    @memoized_property
    def speed_max(self):
        return self.speed_lowpass.max(initial=0)

    @property
    def speed_lowpass(self):
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(np.linalg.norm(self.velocities, axis=1), size=40)

    @memoized_property
    def direction_mean(self):
        """En grados de la horizontal, [0, 360)"""
        v = self.velocity_mean
        d = np.rad2deg(np.arctan2(v[1], v[0]))
        return (d + 360) % 360

    @property
    def velocities(self):
        if len(self.blobs) < 2:
            return np.array([[0, 0]])
        return np.array(np.diff(self.path()[:, 0:2], axis=0))

    @property
    def areas(self):
        return np.array([blob.area for frame, blob in self.blobs.items()])

    @memoized_property
    def area_mean(self):
        return np.mean(self.areas)

    @memoized_property
    def area_median(self):
        return np.median(self.areas)

    @memoized_property
    def width_mean(self):
        return np.mean([blob.width for frame, blob in self.blobs.items()])

    @memoized_property
    def length_mean(self):
        return np.mean([blob.length for frame, blob in self.blobs.items()])

    @memoized_property
    def width_median(self):
        return np.median([blob.width for frame, blob in self.blobs.items()])

    @memoized_property
    def length_median(self):
        return np.median([blob.length for frame, blob in self.blobs.items()])

    def first_frame(self) -> Optional[FrameNumber]:
        if len(self.blobs) == 0: return None
        return min(self.blobs.keys())

    def last_frame(self) -> Optional[FrameNumber]:
        if len(self.blobs) == 0: return None
        return max(self.blobs.keys())

    def first_blob(self) -> Blob:
        return self.at(self.first_frame())

    def last_blob(self) -> Blob:
        return self.at(self.last_frame())

    def cut(self, last: FrameNumber, first: FrameNumber = 0) -> 'Track':
        return Track(self.id, {frame: blob for frame, blob in self.blobs.items() if first < frame < last})

    def get_safe_blobs(self, percentage) -> Dict[FrameNumber, Blob]:
        """Get blobs that are at least ``percentage`` of the imshape into the frame"""
        return {frame: blob for frame, blob in self.blobs.items() if blob.is_fully_visible(percentage)}

    def as_closed(self) -> 'Track':
        return self

    # region Drawing

    def draw_track_line(self, frame: FrameNumber, image: ColorImage, last_n_frames=10) -> ColorImage:
        from .common import draw_line
        copy = image.copy()
        frames_involved = self.blobs.keys()
        first_frame = max(frame - last_n_frames, min(frames_involved))
        last_frame = min(frame, max(frames_involved))
        frames_to_draw = range(first_frame + 1, last_frame + 1)
        for f in frames_to_draw:
            center1 = self.at(f - 1).center_xy
            center2 = self.at(f).center_xy
            copy = draw_line(copy, center1, center2, color=self.color, width=3)
        return copy

    def draw_blob(self, frame: FrameNumber, image: ColorImage, label_color: Color = None) -> ColorImage:
        from .common import Colors
        blob = self.at(frame)
        if blob is None:
            return image
        label_color = label_color or Colors.BLACK
        return blob.draw_contour(image, text=f'T{self.id}', color=self.color, text_color=label_color)

    T = TypeVar('T', bound='Track')

    @classmethod
    def draw_tracks(cls, tracks: List[T], image: ColorImage, frame: FrameNumber) -> ColorImage:
        """Returns a copy of ``image`` with all blobs in ``frame`` of ``tracks`` drawn onto it"""
        copy = image.copy()
        for track in tracks:
            copy = cls.draw_blob(track, frame, copy)
        return copy

    # endregion

    @classmethod
    def get(cls: Type[T], tracks: List[T], track_id: TrackId) -> T:
        return [track for track in tracks if track.id == track_id][0]

    # region Serialization

    class Serial(TypedDict):
        id: int
        loaded: Optional[bool]
        load_probability: Optional[float]
        blobs: Dict[str, Blob.Serial]

    def encode(self) -> 'Track.Serial':
        return {
            "id":               self.id,
            "loaded":           self.loaded.to_bool(),
            "load_probability": self.load_probability,
            "blobs":            {str(i): blob.encode() for i, blob in self.blobs.items()}
        }

    @classmethod
    def decode(cls, serial: 'Track.Serial', imshape: Tuple[int, int]):
        self = cls(
            TrackId(serial["id"]),
            {int(i): Blob.decode(blob, imshape) for i, blob in serial["blobs"].items()},
            force_load_to=serial.get('loaded', None)
        )
        if 'load_probability' in serial:
            self.set_load_probability(serial['load_probability'])
        return self

    # endregion
