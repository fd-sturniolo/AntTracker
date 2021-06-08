import numpy as np
import pims
import cv2 as cv
import skimage.draw as skdraw
import skimage.feature as skfeature
import skimage.filters as skfilters
import skimage.measure as skmeasure
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import ujson
from scipy.spatial import cKDTree
from typing import Any, Dict, Generator, List, Tuple, TypedDict, Sequence

from .blob import Blob
from .common import BinaryMask, ColorImage, GrayscaleImage, ProgressBar, Video, rgb2gray, to_json, FrameNumber, \
    eq_gen_it
from .parameters import SegmenterParameters, LogWSegmenterParameters, DohSegmenterParameters


Blobs = List[Blob]

def _get_mask(frame: GrayscaleImage, last_frames: List[GrayscaleImage], *, params: SegmenterParameters):
    if len(last_frames) == 0:
        background = GrayscaleImage(np.zeros_like(frame))
    else:
        background = GrayscaleImage(np.median(last_frames, axis=0))

    movement: GrayscaleImage = np.abs(frame - background)

    mask: BinaryMask = movement > params.movement_detection_threshold

    # Descartar la máscara si está llena de movimiento (se movió la cámara!)
    if np.count_nonzero(mask) > np.size(mask) * params.discard_percentage:
        return np.zeros(mask.shape, dtype='bool')

    radius = params.minimum_ant_radius

    mask = cv.morphologyEx(mask.astype('uint8'), cv.MORPH_CLOSE, skmorph.disk(round(radius)))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, skmorph.disk(round(radius * 0.8)))
    mask = cv.dilate(mask, skmorph.disk(round(radius)))

    return mask

def _get_blobs_in_frame_with_steps_logw(frame: GrayscaleImage, movement_mask: BinaryMask, params: SegmenterParameters,
                                        prev_blobs: List[Blob]):
    def empty():
        return [], np.zeros_like(frame, dtype=float), \
               np.zeros_like(frame, dtype=float), \
               np.zeros_like(frame, dtype=float), \
               np.zeros_like(frame, dtype=bool), \
               np.zeros_like(frame, dtype='uint8')

    if not movement_mask.any():
        return empty()
    gauss = skfilters.gaussian(frame, sigma=params.gaussian_sigma)
    log = skfilters.laplace(gauss, mask=movement_mask)

    if not log.any():
        return empty()

    try:
        t = skfilters.threshold_isodata(log)
    except IndexError:
        print("Umbralizado fallido (no había bordes significativos en las regiones en movimiento). Salteando frame")
        return empty()
    threshed_log = log.copy()
    threshed_log[threshed_log > t] = 0

    intensity_mask = threshed_log.copy()
    intensity_mask[intensity_mask != 0] = True
    intensity_mask[intensity_mask == 0] = False

    blobs: Blobs = []
    # region Watershed if there were blobs too close to eachother in last frame
    intersection_zone = np.zeros_like(frame, dtype='bool')
    close_markers_labels = np.zeros_like(frame, dtype='uint8')
    if len(prev_blobs) > 1:
        points = np.array([[blob.center_xy.y, blob.center_xy.x] for blob in prev_blobs])
        kdt = cKDTree(points)
        idx = []
        # every blob (kdt) against every other blob (points[i])
        for i, blob in enumerate(prev_blobs[:-1]):
            new = list(kdt.query_ball_point(points[i], maximum_clear_radius(blob.radius)))
            if len(new) == 1:
                new.remove(i)
            idx.extend(new)
        close_idx = np.unique(idx)
        if len(close_idx) > 0:
            close_markers_mask = np.zeros_like(frame, dtype='uint8')
            for idx in close_idx:
                rr, cc = skdraw.disk((points[idx][0], points[idx][1]),
                                     int(maximum_clear_radius(prev_blobs[idx].radius)),
                                     shape=intersection_zone.shape)
                intersection_zone[rr, cc] = True
                rr, cc = skdraw.disk((points[idx][0], points[idx][1]), int(params.minimum_ant_radius * 1.5),
                                     shape=intersection_zone.shape)
                close_markers_mask[rr, cc] = idx + 1
            close_markers_labels = skseg.watershed(np.zeros_like(frame), markers=close_markers_mask,
                                                   mask=intensity_mask * intersection_zone)
            props = skmeasure.regionprops(close_markers_labels)
            for p in props:
                if p.area < minimum_ant_area(params.minimum_ant_radius):
                    continue
                label: bool = p.label
                blobs.append(Blob(imshape=frame.shape, mask=(close_markers_labels == label),
                                  approx_tolerance=params.approx_tolerance))
    # endregion

    mask_not_intersecting = intensity_mask * (~intersection_zone)
    labels, nlabels = skmeasure.label(mask_not_intersecting, return_num=True)
    props = skmeasure.regionprops(labels)

    for p in props:
        if p.area < minimum_ant_area(params.minimum_ant_radius):
            continue
        label: bool = p.label
        blobs.append(Blob(imshape=frame.shape, mask=(labels == label), approx_tolerance=params.approx_tolerance))

    return blobs, gauss, log, threshed_log, intersection_zone, (close_markers_labels + labels)

def _get_blobs_in_frame_with_steps_doh(frame: GrayscaleImage, movement_mask: BinaryMask, params: SegmenterParameters):
    if not movement_mask.any():
        return [], np.zeros_like(frame, dtype=float), \
               np.zeros_like(frame, dtype=float), \
               np.zeros_like(frame, dtype='uint8'), \
               np.zeros_like(frame, dtype='uint8'),
    masked_frame = frame.copy()
    masked_frame[~movement_mask] = 255
    yxs: np.ndarray = skfeature.blob_doh(masked_frame, min_sigma=params.doh_min_sigma, max_sigma=params.doh_max_sigma,
                                         num_sigma=params.doh_num_sigma)
    markers = yxs[:, 0:2].astype(int).tolist()

    marker_mask = np.zeros_like(frame, dtype='uint8')
    for _id, marker in enumerate(markers, 1):
        if not movement_mask[marker[0], marker[1]]:
            continue
        marker_mask[marker[0], marker[1]] = _id
        # rr, cc = skdraw.circle_perimeter(marker[0], marker[1], 10, shape=masked_frame.shape)
        # masked_frame[rr, cc] = 0

    gauss = skfilters.gaussian(frame, sigma=params.gaussian_sigma)
    log = skfilters.laplace(gauss, mask=movement_mask)

    t = skfilters.threshold_isodata(log)
    labels = skseg.watershed(log, markers=marker_mask, mask=(log < t))
    props = skmeasure.regionprops(labels)

    blobs: Blobs = []
    for p in props:
        if p.area < minimum_ant_area(params.minimum_ant_radius):
            continue
        label: bool = p.label
        blobs.append(Blob(imshape=frame.shape, mask=(labels == label), approx_tolerance=params.approx_tolerance))

    return blobs, gauss, log, labels, masked_frame

def _get_blobs_logw(frame: GrayscaleImage, movement_mask: BinaryMask, params: SegmenterParameters,
                    prev_blobs: List[Blob]):
    return _get_blobs_in_frame_with_steps_logw(frame, movement_mask, params, prev_blobs)[0]

def _get_blobs_doh(frame: GrayscaleImage, movement_mask: BinaryMask, params: SegmenterParameters):
    return _get_blobs_in_frame_with_steps_doh(frame, movement_mask, params)[0]

def minimum_ant_area(min_radius):
    return np.pi * min_radius ** 2

def maximum_clear_radius(radius):
    return radius * 3

class Segmenter:
    def __init__(self, video: Video = None, params: SegmenterParameters = None):
        if video is None:
            return
        self.__frames_with_blobs: Dict[FrameNumber, Blobs] = {}
        self.__last_frames = []
        self.__video = video
        self.__prev_blobs: Tuple[FrameNumber, Blobs] = (-1, [])
        self.params = params
        self.video_length = len(video)
        self.video_shape = tuple(video[0].shape[0:2])

    @property
    def is_finished(self):
        if not self.__frames_with_blobs: return False
        segmented_all_frames = eq_gen_it(self.__frames_with_blobs.keys(), range(self.video_length))
        return segmented_all_frames

    @property
    def reached_last_frame(self):
        return bool(self.__frames_with_blobs) and max(self.__frames_with_blobs) == self.video_length - 1

    @classmethod
    def segment_single(cls, params: SegmenterParameters, frame: np.ndarray, previous_frames: Sequence[ColorImage]):
        gray_frame = rgb2gray(frame)
        pvf = [rgb2gray(p) for p in previous_frames]
        mask = _get_mask(gray_frame, pvf, params=params)
        mock = cls()
        mock.params = params
        return cls._get_blobs(mock, gray_frame, mask, [])

    def _get_mask(self, frame):
        return _get_mask(frame, self.__last_frames, params=self.params)

    def _get_blobs(self, gray_frame, mask, prev_blobs) -> Blobs:
        raise NotImplementedError

    def __cycle_last_frames(self, frame: GrayscaleImage):
        if len(self.__last_frames) < self.params.movement_detection_history:
            self.__last_frames.append(frame)
        else:
            self.__last_frames[:-1] = self.__last_frames[1:]
            self.__last_frames[-1] = frame

    def segment_rolling_continue(self):
        """Continuar segmentando desde el último frame segmentado."""
        yield from self.segment_rolling_from(max(self.__frames_with_blobs) + 1)

    def segment_rolling_from(self, from_frame_n: FrameNumber, prev_blobs: Blobs = None):
        """Segmentar desde un frame en particular. Bajo operación normal, ``prev_blobs`` debe ser None.
        Pueden ocurrir estas situaciones:

        - Se comienza desde el frame 0: no hay blobs previos.
        - Se comienza desde un frame segmentado o el siguiente al último: Usa los blobs en ``__frames_with_blobs``.
          Ver también: ``segment_rolling_continue()``
        - Se comienza desde algún otro frame en particular: se deberá proporcionar la lista de blobs previos,
          probablemente de una instancia de ``tracking.Tracker``. De no tener este dato, puede pasarse una lista vacía,
          a riesgo de perder reproducibilidad.
        """
        if self.__video is None:
            raise ValueError("Este segmentador no tiene un video cargado. Use set_video()")
        if from_frame_n < 0 or from_frame_n >= self.video_length:
            raise ValueError(f"Frame {from_frame_n} inexistente")
        if from_frame_n == 0:
            prev_blobs = []
        elif self.__frames_with_blobs and (from_frame_n - 1) in self.__frames_with_blobs:
            prev_blobs = self.__frames_with_blobs[from_frame_n - 1]
        elif prev_blobs is None:
            raise ValueError(f"Debe proporcionar los blobs del frame anterior ({from_frame_n - 1}): "
                             "puede obtenerlos a partir del Tracker "
                             "si se dispone de él, o bien proporcionar una lista vacía "
                             "(corriendo el riesgo de perder reproducibilidad)")
        n = min(self.params.movement_detection_history, from_frame_n)
        self.__last_frames = [rgb2gray(f) for f in self.__video[from_frame_n - n:from_frame_n]] if from_frame_n != 0 else []
        for frame_n, frame in enumerate(self.__video[from_frame_n:], from_frame_n):
            if frame_n in self.__frames_with_blobs:
                yield frame_n, self.__frames_with_blobs[frame_n]
            else:
                gray_frame = rgb2gray(frame)
                mask = self._get_mask(gray_frame)
                blobs = self._get_blobs(gray_frame, mask, prev_blobs)
                self.__cycle_last_frames(gray_frame)
                self.__frames_with_blobs[frame_n] = blobs
                self.__prev_blobs = (frame_n, blobs)
                prev_blobs = blobs
                yield frame_n, blobs

    @property
    def frames_with_blobs(self) -> Generator[Tuple[FrameNumber, Blobs], None, None]:
        if self.is_finished:
            yield from self.__frames_with_blobs.items()
            return
        yield from self.segment_rolling_from(0)

    def blobs_at(self, frame_n: FrameNumber):
        if frame_n in self.__frames_with_blobs:
            return self.__frames_with_blobs[frame_n]
        else:
            raise KeyError("Ese cuadro no fue segmentado aún")

    def set_video(self, video: Video):
        """Usar luego de cargar un segmentador serializado."""
        self.__video = video

    class Serial(TypedDict):
        frames_with_blobs: Dict[FrameNumber, List[Blob.Serial]]
        parameters: Dict[str, Any]
        video_length: int
        video_shape: Tuple[int, int]

    def encode(self):
        return {
            'frames_with_blobs': {str(frame): [blob.encode() for blob in blobs] for frame, blobs in
                                  self.__frames_with_blobs.items()},
            'parameters':        dict(self.params.items()),
            'video_length':      self.video_length,
            'video_shape':       self.video_shape,
        }

    @classmethod
    def decode(cls, d: 'Segmenter.Serial'):
        segmenter = cls()
        shape = d['video_shape']
        segmenter.__frames_with_blobs = {
            FrameNumber(frame): [Blob.decode(blob, shape) for blob in blobs]
            for frame, blobs in d['frames_with_blobs'].items()
        }
        segmenter.params = SegmenterParameters(d['parameters'])
        segmenter.video_length = d['video_length']
        segmenter.video_shape = shape
        return segmenter

    def serialize(self) -> str:
        return to_json(self.encode())

    @classmethod
    def deserialize(cls, *, filename=None, jsonstring=None):
        if filename is not None:
            with open(filename, 'r') as file:
                segmenter_dict = ujson.load(file)
        elif jsonstring is not None:
            segmenter_dict = ujson.loads(jsonstring)
        else:
            raise TypeError("Provide either JSON string or filename.")
        return cls.decode(segmenter_dict)

class LogWSegmenter(Segmenter):
    def __init__(self, video: Video = None, params: SegmenterParameters = None):
        if params is None: params = LogWSegmenterParameters()
        super(LogWSegmenter, self).__init__(video, params)

    def _get_blobs(self, gray_frame, mask, prev_blobs):
        return _get_blobs_logw(gray_frame, mask, self.params, prev_blobs)

class DohSegmenter(Segmenter):
    def __init__(self, video: Video = None, params: SegmenterParameters = None):
        if params is None: params = DohSegmenterParameters()
        super(DohSegmenter, self).__init__(video, params)

    def _get_blobs(self, gray_frame, mask, prev_blobs):
        return _get_blobs_doh(gray_frame, mask, self.params)
