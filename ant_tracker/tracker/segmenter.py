import numpy as np
import pims
import skimage.draw as skdraw
import skimage.feature as skfeature
import skimage.filters as skfilters
import skimage.measure as skmeasure
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import ujson
from packaging.version import Version
from scipy.spatial import cKDTree
from typing import Any, Dict, Generator, List, Tuple, TypedDict, Sequence

from .blob import Blob
from .common import BinaryMask, ColorImage, GrayscaleImage, ProgressBar, Video, rgb2gray, to_json, FrameNumber, \
    eq_gen_it
from .parameters import SegmenterParameters, LogWSegmenterParameters, DohSegmenterParameters

SegmenterVersion = Version("2.0.2dev1")

Blobs = List[Blob]

def _get_mask_with_steps(frame: GrayscaleImage, last_frames: List[GrayscaleImage], *, params: SegmenterParameters):
    if len(last_frames) == 0:
        background = GrayscaleImage(np.zeros_like(frame))
    else:
        background = GrayscaleImage(np.median(last_frames, axis=0))

    movement: GrayscaleImage = np.abs(frame - background)

    mask: BinaryMask = movement > params.movement_detection_threshold

    # Descartar la máscara si está llena de movimiento (se movió la cámara!)
    if np.count_nonzero(mask) > np.size(mask) * params.discard_percentage:
        zeros = np.zeros(mask.shape, dtype='bool')
        return zeros, zeros, zeros, zeros, background, movement

    radius = params.minimum_ant_radius
    closed_mask = skmorph.binary_closing(mask, skmorph.disk(round(radius)))
    opened_mask = skmorph.binary_opening(closed_mask, skmorph.disk(round(radius * 0.8)))
    dilated_mask = skmorph.binary_dilation(opened_mask, skmorph.disk(round(radius)))

    return dilated_mask, mask, closed_mask, opened_mask, background, movement

def _get_mask(frame: GrayscaleImage, last_frames: List[GrayscaleImage], *, params: SegmenterParameters):
    return _get_mask_with_steps(frame, last_frames, params=params)[0]

def _get_blobs_in_frame_with_steps_logw(frame: GrayscaleImage, movement_mask: BinaryMask, params: SegmenterParameters,
                                        prev_blobs: List[Blob]):
    if not movement_mask.any():
        return [], np.zeros_like(frame, dtype=float), \
               np.zeros_like(frame, dtype=float), \
               np.zeros_like(frame, dtype=float), \
               np.zeros_like(frame, dtype=bool), \
               np.zeros_like(frame, dtype='uint8'),
    gauss = skfilters.gaussian(frame, sigma=params.gaussian_sigma)
    log = skfilters.laplace(gauss, mask=movement_mask)

    t = skfilters.threshold_isodata(log)
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
        segmented_all_frames = eq_gen_it(self.__frames_with_blobs.values(), range(self.video_length))
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

    def segment_rolling_from(self, frame_n: FrameNumber, prev_blobs: Blobs = None):
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
        if frame_n < 0 or frame_n >= self.video_length:
            raise ValueError(f"Frame {frame_n} inexistente")
        if frame_n == 0:
            prev_blobs = []
        elif self.__frames_with_blobs and (frame_n - 1) in self.__frames_with_blobs:
            prev_blobs = self.__frames_with_blobs[frame_n - 1]
        elif prev_blobs is None:
            raise ValueError(f"Debe proporcionar los blobs del frame anterior ({frame_n - 1}): "
                             "puede obtenerlos a partir del Tracker "
                             "si se dispone de él, o bien proporcionar una lista vacía "
                             "(corriendo el riesgo de perder reproducibilidad)")
        n = min(self.params.movement_detection_history, frame_n)
        self.__last_frames = [rgb2gray(f) for f in self.__video[frame_n - n:frame_n]] if frame_n != 0 else []
        for frame_n, frame in enumerate(self.__video[frame_n:], frame_n):
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

    @property
    def version(self):
        raise NotImplementedError

    class Serial(TypedDict):
        frames_with_blobs: Dict[FrameNumber, List[Blob.Serial]]
        parameters: Dict[str, Any]
        video_length: int
        video_shape: Tuple[int, int]
        version: str

    def encode(self):
        return {
            'frames_with_blobs': {str(frame): [blob.encode() for blob in blobs] for frame, blobs in
                                  self.__frames_with_blobs.items()},
            'parameters':        dict(self.params.items()),
            'video_length':      self.video_length,
            'video_shape':       self.video_shape,
            'version':           str(self.version)
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

    @property
    def version(self):
        return Version('2.0.2dev1')

    def _get_blobs(self, gray_frame, mask, prev_blobs):
        return _get_blobs_logw(gray_frame, mask, self.params, prev_blobs)

class DohSegmenter(Segmenter):
    def __init__(self, video: Video = None, params: SegmenterParameters = None):
        if params is None: params = DohSegmenterParameters()
        super(DohSegmenter, self).__init__(video, params)

    @property
    def version(self):
        return Version('2.0.2dev2')

    def _get_blobs(self, gray_frame, mask, prev_blobs):
        return _get_blobs_doh(gray_frame, mask, self.params)

# noinspection PyUnboundLocalVariable
def main():
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.widgets import Slider
    from mpl_toolkits.axes_grid1 import ImageGrid

    from .plotcommon import Animate, PageSlider
    from .common import Colors
    import argparse
    parser = argparse.ArgumentParser(description="Visualize segmentation of a video")
    parser.add_argument('file')
    parser.add_argument('--firstFrame', '-f', type=int, default=None, metavar="F",
                        help="Primer frame a procesar")
    parser.add_argument('--lastFrame', '-l', type=int, default=None, metavar="L",
                        help="Último frame a procesar")
    parser.add_argument('--draw', '-d', type=bool, default=False, metavar="D",
                        help="Mostrar la segmentación en imágenes")
    parser.add_argument('--play', '-p', type=bool, default=False, metavar="P",
                        help="Avanzar frames automáticamente (con --draw)")

    args = parser.parse_args()
    file: str = args.file
    draw = args.draw
    play = args.play

    video = pims.PyAVReaderIndexed(f"{file}")
    p = SegmenterParameters(gaussian_sigma=5)

    frame_n = 0 if args.firstFrame is None else args.firstFrame
    last_frame_to_process = len(video) if args.lastFrame is None else args.lastFrame
    print(f"Processing from frames {frame_n} to {last_frame_to_process}")
    last_drawn_frame_n = -1
    exit_flag = False
    update = False
    update_page = False
    page = 0
    total_pages = 13
    if draw:
        def on_key_press(event):
            nonlocal frame_n, exit_flag, play, page, page_slider, update, update_page
            if event.key == 'a':
                frame_n -= 1
            elif event.key == 'd':
                frame_n += 1
            elif event.key == 'p':
                play = not play
                update = True
            elif event.key == 'k':
                page = (page + 1) % total_pages
                page_slider.set_val(page)
                update_page = True
            elif event.key == 'j':
                page = (page - 1) % total_pages
                page_slider.set_val(page)
                update_page = True
            elif event.key == 't':
                nonlocal log
                try:
                    log
                except UnboundLocalError:
                    print("log undefined")
                    return
                skfilters.try_all_threshold(log)
            elif event.key == 'escape':
                exit_flag = True

        fig: Figure = plt.figure()
        grid = ImageGrid(fig, (0.1, 0.1, 0.8, 0.8),
                         nrows_ncols=(1, 2),
                         share_all=True,
                         axes_pad=0.05,
                         label_mode="1",
                         )
        fig.suptitle(f"{frame_n=}")
        fig.canvas.mpl_connect('key_press_event', on_key_press)

        def __sigma_update_fn(val):
            nonlocal p, update
            p.gaussian_sigma = val
            update = True

        # noinspection PyPep8Naming
        sigmaS = Slider(fig.add_axes([0.1, 0.1, 0.8, 0.04]), 'sigma', 1., 20, valinit=p.gaussian_sigma, valstep=0.2)
        sigmaS.on_changed(__sigma_update_fn)

        ax_page_slider = fig.add_axes([0.1, 0.05, 0.8, 0.04])
        page_slider = PageSlider(ax_page_slider, 'Page', total_pages, activecolor="orange")

        def __page_update_fn(val):
            nonlocal page, update_page
            i = int(val)
            page = i
            update_page = True

        page_slider.on_changed(__page_update_fn)
    else:
        progress_bar = ProgressBar(last_frame_to_process)

    print(f"{exit_flag=}")
    print(f"{frame_n=}")
    print(f"{len(video)=}")
    last_frames = []

    def draw_step(ax_, step):
        Animate.draw(ax_, step['im'], autoscale=step.get('autoscale', False))
        ax_.set_title(step['title'])

    prev_blobs = []
    while not exit_flag and frame_n < last_frame_to_process:
        if last_drawn_frame_n != frame_n or update:
            print(f"{frame_n=}")
            frame: ColorImage = video[frame_n]

            grayframe = rgb2gray(frame)

            if last_drawn_frame_n != frame_n:
                movement_mask, first_mask, closed_mask, opened_mask, background, movement = _get_mask_with_steps(
                    grayframe, last_frames, params=p)
                if len(last_frames) < p.movement_detection_history:
                    last_frames.append(grayframe)
                else:
                    last_frames[:-1] = last_frames[1:]
                    last_frames[-1] = grayframe

            _out = _get_blobs_in_frame_with_steps_logw(grayframe,
                                                       movement_mask,
                                                       params=p,
                                                       prev_blobs=prev_blobs)
            blobs, gauss, log, threshed_log, intersection_zone, labels = _out

            prev_blobs = blobs
            frame_with_blobs = Blob.draw_blobs(blobs, frame).copy()

            if draw:
                for blob in blobs:
                    # only do watershed where previous frame blobs had intersecting circles
                    rr, cc = skdraw.circle_perimeter(blob.center_xy[1], blob.center_xy[0],
                                                     int(p.minimum_ant_radius),
                                                     shape=frame_with_blobs.shape)
                    frame_with_blobs[rr, cc] = Colors.RED
                    rr, cc = skdraw.circle_perimeter(blob.center_xy[1], blob.center_xy[0],
                                                     int(maximum_clear_radius(blob.radius)),
                                                     shape=frame_with_blobs.shape)
                    frame_with_blobs[rr, cc] = Colors.BLUE

                fig.suptitle(f"{frame_n=}")
                last_drawn_frame_n = frame_n
                if play:
                    frame_n += 1
                update_page = True
                update = False
            else:
                frame_n += 1
                progress_bar.next()
        if draw:
            if update_page:
                steps = [
                    {'im': frame, 'title': "frame"},
                    {'im': movement, 'title': "movement", 'autoscale': True},
                    {'im': first_mask, 'title': "first_mask"},
                    {'im': closed_mask, 'title': "closed_mask"},
                    {'im': opened_mask, 'title': "opened_mask"},
                    {'im': movement_mask, 'title': "dilated_mask"},
                    {'im': frame, 'title': "frame"},
                    {'im': gauss, 'title': "gauss", 'autoscale': True},
                    {'im': log, 'title': "log", 'autoscale': True},
                    {'im': threshed_log, 'title': "threshed_log", 'autoscale': True},
                    {'im': intersection_zone, 'title': "intersection_zone"},
                    {'im': labels, 'title': "labels", 'autoscale': True},
                    {'im': frame_with_blobs, 'title': "blobs"},
                ]

                draw_step(grid[0], steps[page])
                draw_step(grid[1], steps[(page + 1) % total_pages])

                plt.draw()
                update_page = False
            plt.pause(0.05)
    plt.close()

if __name__ == '__main__':
    main()
