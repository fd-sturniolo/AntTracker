import dataclasses
from dataclasses import InitVar, dataclass, field

import datetime
import numpy as np
import ujson
from packaging.version import Version
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, TypedDict, Union, Callable

from .common import FrameNumber, to_json, Side, Rect, filehash, ensure_path, SerializableEnum
from .parameters import SegmenterParameters, TrackerParameters
from .track import Track
from ..version import __version__

class Direction(SerializableEnum):
    EN = "EN"
    "Entrando al nido"
    SN = "SN"
    "Saliendo del nido"
    UN = "??"
    "Desconocido/Irrelevante"

@dataclass
class TracksInfo:
    tracks: List[Track] = field(repr=False)

    #! cuidado: cuando se resuelva #8 puede que haya
    #! un problema con este campo por tener el mismo nombre
    version: Version

    segmenter_parameters: SegmenterParameters
    tracker_parameters: TrackerParameters

    video_name: str = field(init=False)
    video_hash: str = field(init=False)
    video_shape: Tuple[int, int] = field(init=False)
    video_length: int = field(init=False)
    video_fps_average: float = field(init=False)
    video_path: InitVar[Optional[Union[Path, str]]] = None

    file_extension: ClassVar[str] = '.trk'

    def __post_init__(self, video_path: Optional[Union[Path, str]] = None):
        if video_path is not None:
            if not isinstance(video_path, Path):
                video_path = Path(video_path)
            self.video_name = video_path.name
            try:
                import av
                container = av.open(str(video_path))  # noqa
                self.video_length = container.streams.video[0].frames
                self.video_shape = (container.streams.video[0].codec_context.height,
                                    container.streams.video[0].codec_context.width)
                self.video_fps_average = float(container.streams.video[0].average_rate)

            except ImportError:
                try:
                    import cv2 as cv
                    video = cv.VideoCapture(str(video_path), cv.CAP_FFMPEG)
                    self.video_shape = (int(video.get(cv.CAP_PROP_FRAME_HEIGHT)),
                                        int(video.get(cv.CAP_PROP_FRAME_WIDTH)))
                    self.video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
                    self.video_fps_average = video.get(cv.CAP_PROP_FPS)
                except ImportError:
                    raise ImportError("No se encontraron las librerías cv2 ni av")
            self.video_hash = filehash(video_path)

    @property
    def video_duration(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=float(self.video_length / self.video_fps_average))

    def filter_tracks(self, *,
                      last_frame: FrameNumber = None,
                      length_of_tracks=6,
                      filter_center_center=True
                      ) -> List[Track]:
        """Returns a filtered ``self.tracks``. If a track has blobs beyond ``last_frame`` it is cut short.

        Parameters:
            last_frame: Last frame to consider.
            length_of_tracks: Minimum number of blobs in a track.
            filter_center_center: Remove tracks that started and ended in the center.
        Returns:
            The filtered list of Tracks.
        """
        _filter = self.filter_func(last_frame=last_frame,
                                   length_of_tracks=length_of_tracks,
                                   filter_center_center=filter_center_center)
        return [
            track if last_frame is None else track.cut(last_frame)
            for track in self.tracks if _filter(track)
        ]

    def filter_func(self, *,
                    last_frame: FrameNumber = None,
                    length_of_tracks=6,
                    filter_center_center=True) -> Callable[[Track], bool]:
        """
        Parameters:
            last_frame: Last frame to consider.
            length_of_tracks: Minimum number of blobs in a track.
            filter_center_center: Remove tracks that started and ended in the center.
        Returns:
            A filtering function for Tracks
        """

        def _filter(track: Track):
            if len(track.blobs) < length_of_tracks: return False
            if filter_center_center:
                if track.direction_of_travel(self.video_shape) == (Side.Center, Side.Center):
                    return False
            if last_frame is not None:
                if track.last_frame() > last_frame: return False
            return True

        return _filter

    def last_tracked_frame(self) -> FrameNumber:
        return max([track.last_frame() for track in self.tracks])

    def get_blobs_in_frame(self, frame: FrameNumber):
        blobs = []
        for track in self.tracks:
            blob = track.at(frame)
            if blob is None: continue
            blobs.append(blob)
        return blobs

    def track_direction(self, track: Track, nest_side: Side):
        _from, to = track.direction_of_travel(self.video_shape)
        if _from != nest_side and to == nest_side:
            return Direction.EN
        elif _from == nest_side and to != nest_side:
            return Direction.SN
        else:
            return Direction.UN

    def is_from_video(self, path: Union[Path, str]):
        path = ensure_path(path)
        return self.video_hash == filehash(path)

    # region Serialization

    class Serial(TypedDict):
        tracks: List[Track.Serial]

        version: str
        tracker_parameters: Dict[str, Any]
        segmenter_parameters: Dict[str, Any]
        video_name: str
        video_hash: str
        video_length: int
        video_shape: Tuple[int, int]
        video_fps_average: float

    def encode(self) -> 'TracksInfo.Serial':
        return {
            "tracks":               [track.encode() for track in self.tracks],
            "version":              str(self.version),
            "tracker_parameters":   dataclasses.asdict(self.tracker_parameters),
            "segmenter_parameters": dataclasses.asdict(self.segmenter_parameters),
            "video_name":           self.video_name,
            "video_hash":           self.video_hash,
            "video_length":         self.video_length,
            "video_shape":          self.video_shape,
            "video_fps_average":    self.video_fps_average,
        }

    @classmethod
    def decode(cls, serial: 'TracksInfo.Serial'):
        shape = tuple(serial["video_shape"])
        self = TracksInfo(
            tracks=[Track.decode(track, shape) for track in serial["tracks"]],
            version=Version(serial.get('version',"1.0.0")),
            segmenter_parameters=SegmenterParameters(serial["segmenter_parameters"]),
            tracker_parameters=TrackerParameters(serial["tracker_parameters"]),
        )
        self.video_name = serial["video_name"]
        self.video_hash = serial["video_hash"]
        self.video_shape = shape
        self.video_length = serial["video_length"]
        #TODO: #9
        if "video_fps_average" in serial:
            self.video_fps_average = serial["video_fps_average"]
        else:
            # safeguard for AntLabeler version <=1.6
            self.video_fps_average = None
        self.__class__ = cls
        return self

    def serialize(self) -> str:
        return to_json(self.encode())

    @classmethod
    def deserialize(cls, *, filename=None, jsonstring=None):
        if filename is not None:
            with open(filename, 'r') as file:
                d = ujson.load(file)
        elif jsonstring is not None:
            d = ujson.loads(jsonstring)
        else:
            raise TypeError("Provide either JSON string or filename.")
        return cls.decode(d)

    @classmethod
    def _is_extension_valid(cls, file: Path):
        return file.suffix == cls.file_extension

    def save(self, file: Union[Path, str]):
        file = ensure_path(file)
        if not self._is_extension_valid(file):
            raise ValueError(f'Wrong extension ({file.suffix}). Only {self.file_extension} files are valid.')
        with file.open('w') as f:
            f.write(self.serialize())

    @classmethod
    def load(cls, file: Union[Path, str]):
        file = ensure_path(file)
        if not cls._is_extension_valid(file):
            raise ValueError(f'Wrong extension ({file.suffix}). Only {cls.file_extension} files are valid.')
        return cls.deserialize(filename=file)

    # endregion

def reposition_into_crop(info: TracksInfo, crop_rect: Rect):
    from .blob import Blob
    from .track import TrackId

    def _clip_contour(contour, imshape):
        def consecutive_not_equal(array):
            return np.append(np.where(np.diff(array) != 0), len(array) - 1)

        # if the contour is completely outside of imshape
        if np.all(contour[:, 0] < 0) or np.all(contour[:, 0] > imshape[0] - 1) or \
                np.all(contour[:, 1] < 0) or np.all(contour[:, 1] > imshape[1] - 1):
            return None
        contour[:, 0] = np.clip(contour[:, 0], 0, imshape[0] - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, imshape[1] - 1)
        relevant_y_idx = consecutive_not_equal(contour[:, 0])
        relevant_x_idx = consecutive_not_equal(contour[:, 1])
        relevant_idx = np.intersect1d(relevant_x_idx, relevant_y_idx)
        contour = contour[relevant_idx, :]

        return contour

    offx, offy = crop_rect.topleft.x, crop_rect.topleft.y
    new_shape = crop_rect.height, crop_rect.width
    new_tracks = []
    new_track_id = 0
    for track in info.tracks:
        # a single track may look like multiple tracks if it crosses the crop border
        new_tracks_from_single_track = []
        blobs = {}
        for frame, blob in track.blobs.items():
            new_contour = blob.contour.copy()
            new_contour[:, 0] = blob.contour[:, 0] - offy
            new_contour[:, 1] = blob.contour[:, 1] - offx
            new_contour = _clip_contour(new_contour, new_shape)
            if new_contour is None:
                if blobs:
                    new_tracks_from_single_track.append(
                        Track(TrackId(new_track_id), blobs, force_load_to=track.loaded.to_bool())
                    )
                    new_track_id += 1
                    blobs = {}
            else:
                blobs[frame] = Blob(imshape=new_shape, contour=new_contour)
        if blobs:
            new_tracks_from_single_track.append(
                Track(TrackId(new_track_id), blobs, force_load_to=track.loaded.to_bool())
            )
            new_track_id += 1
        new_tracks.extend(new_tracks_from_single_track)
    info.tracks = new_tracks
    info.video_shape = new_shape
    return info

class TracksCompleteInfo(TracksInfo):
    mm_per_pixel: float = field(init=False)
    crop_rect: Rect = field(init=False)
    nest_side: Side = field(init=False)
    start_time: datetime.datetime = field(init=False)
    end_time: datetime.datetime = field(init=False)

    def __init__(self, info: TracksInfo, mm_per_pixel: float, crop_rect: Rect, nest_side: Side,
                 start_time: datetime.datetime):
        self.tracks = info.tracks

        self.version = info.version
        self.segmenter_parameters = info.segmenter_parameters
        self.tracker_parameters = info.tracker_parameters

        self.video_name = info.video_name
        self.video_hash = info.video_hash
        self.video_length = info.video_length
        self.video_fps_average = info.video_fps_average

        self.mm_per_pixel = mm_per_pixel
        self.crop_rect = crop_rect
        self.nest_side = nest_side

        self.start_time = start_time
        self.end_time = start_time + info.video_duration

        self.video_shape = crop_rect.height, crop_rect.width

    def time_at(self, frame: FrameNumber) -> datetime.datetime:
        if frame < 0: raise ValueError(f"frame < 0")
        if frame >= self.video_length: raise ValueError(f"frame >= length")
        return self.start_time + datetime.timedelta(seconds=frame / self.video_fps_average)

    class __NonFrame(SerializableEnum):
        """For use by ``frame_at`` and ``tracks_in_time``"""
        Before = "Before this video"
        After = "After this video"

    def __frame_at(self, time: datetime.datetime) -> Union[FrameNumber, __NonFrame]:
        """Approximates the frame in the video corresponding to the `time` given.
        If no such frame exists, returns a ``__NonFrame``."""
        if time < self.start_time: return self.__NonFrame.Before
        if time > self.end_time: return self.__NonFrame.After
        delta = time - self.start_time
        return int(delta.seconds * self.video_fps_average)

    def frame_at(self, time: datetime.datetime) -> Optional[FrameNumber]:
        at = self.__frame_at(time)
        if at in self.__NonFrame: return None
        return at

    def tracks_in_time(self, start: datetime.datetime, end: datetime.datetime):
        sf = self.__frame_at(start)
        ef = self.__frame_at(end)

        NF = self.__NonFrame
        if sf == NF.Before:
            sf = 0
        if ef == NF.After:
            sf = self.video_length

        if sf == NF.After or ef == NF.Before:
            return []

        return [track for track in self.tracks if sf <= track.first_frame() <= ef or sf <= track.last_frame() <= ef]

    def tracks_at(self, time: datetime.datetime) -> List[Track]:
        frame = self.frame_at(time)
        if frame is None: return []
        return [track for track in self.tracks if track.at(frame) is not None]

    def track_direction(self, track: Track, nest_side=None):
        """Uses ``self.nest_side``. If nest_side is passed, it is ignored."""
        return super(TracksCompleteInfo, self).track_direction(track, self.nest_side)

    # region Serialization

    class Serial(TracksInfo.Serial):
        mm_per_pixel: float
        crop_rect: Rect
        nest_side: str
        start_time: str
        end_time: str

    def encode(self) -> 'TracksCompleteInfo.Serial':
        d: 'TracksCompleteInfo.Serial' = super(TracksCompleteInfo, self).encode() # noqa
        return {**d, # noqa
                'mm_per_pixel': self.mm_per_pixel,
                'crop_rect':    self.crop_rect,
                'nest_side':    self.nest_side.name,
                'start_time':   self.start_time.isoformat(),
                'end_time':     self.end_time.isoformat(),
                }

    @classmethod
    def decode(cls, serial: 'TracksCompleteInfo.Serial'):
        self: 'TracksCompleteInfo' = super(TracksCompleteInfo, cls).decode(serial) # noqa
        self.mm_per_pixel = serial['mm_per_pixel']
        self.crop_rect = Rect(*serial['crop_rect'])
        self.nest_side = Side[serial['nest_side']]
        self.start_time = datetime.datetime.fromisoformat(serial['start_time'])
        self.end_time = datetime.datetime.fromisoformat(serial['end_time'])
        return self

    # endregion
