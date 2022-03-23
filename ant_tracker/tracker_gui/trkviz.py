from enum import Enum, auto
from multiprocessing import freeze_support

import PySimpleGUI as sg
import math
import matplotlib  # matplotlib is imported by pims by default
import numpy as np
from pathlib import Path
from typing import Union, List

matplotlib.use('agg')  # we set agg to avoid it using tk and risk multithreading issues
import pims
from pims.process import crop

from . import constants as C
from .loading_window import LoadingWindow
from .parameter_extraction import MyGraph
from ..tracker.ant_labeler_info import LabelingInfo
from ..tracker.common import Colors, crop_from_rect, Side, blend
from ..tracker.info import TracksCompleteInfo, TracksInfo
from ..tracker.track import Track, Loaded

class K(Enum):
    Graph = auto()
    FrameSlider = auto()
    FrameBack = auto()
    FrameForw = auto()
    Graph2 = auto()
    FrameSlider2 = auto()
    FrameBack2 = auto()
    FrameForw2 = auto()
    TrackList = auto()
    Tabs = auto()
    Tab1 = auto()
    Tab2 = auto()
    Filter = auto()
    HideMask = auto()

def load_video(path: Union[Path, str]):
    with LoadingWindow("Cargando video..."):
        return pims.PyAVReaderIndexed(path)

def listbox_items(tracks: List[Track], info: TracksInfo):
    return [f"T{t.id} " +
            (f"P:{round(t.load_probability, 2) if t.load_probability else '??'}, " if t.loaded == Loaded.Undefined else
             f"Carga: {'Si' if t.loaded == Loaded.Yes else 'No'}") +
            (f"Dir: {info.track_direction(t)}" if isinstance(info, TracksCompleteInfo) else "")
            for t in tracks]

def pull_track_id(string: str):
    return int(string.split(" ")[0].split("T")[1])


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def trkviz_subprocess(trk_or_tag: Union[Path, str] = None):
    import multiprocessing
    p = multiprocessing.Process(target=trkviz, args=(trk_or_tag,))
    p.start()

def trkviz(trk_or_tag: Union[Path, str] = None):
    try:
        sg.theme(C.THEME)
        if trk_or_tag is None:
            trkfile = sg.popup_get_file("Seleccionar archivo", file_types=(("trk/tag", "*.trk *.tag"),),
                                        no_window=True)
            if not trkfile: return
        else:
            trkfile = trk_or_tag
        trkfile = Path(trkfile)

        complete = False
        label = False
        will_filter = True
        try:
            info = LabelingInfo.load(trkfile)
            label = True
        except:  # noqa
            try:
                info = TracksCompleteInfo.load(trkfile)
                complete = True
            except:  # noqa
                info = TracksInfo.load(trkfile)
        try_this_file = trkfile.parent / info.video_name
        if try_this_file.exists():
            video = load_video(try_this_file)
        else:
            vidfile: str = sg.popup_get_file("Seleccionar video (cancelar para no usar video)",
                                             file_types=(("video", "*.mp4 *.h264 *.avi"),),
                                             no_window=True)
            if vidfile:
                video = load_video(vidfile)
            else:
                video = None
        if video and complete:
            video = crop(video, crop_from_rect(video.frame_shape[0:2], info.crop_rect))
        if label:
            will_filter = False

        with LoadingWindow("Cargando tracks"):
            filtered_tracks = info.filter_tracks(**C.TRACKFILTER)
            if will_filter:
                tracks = filtered_tracks
            else:
                tracks = info.tracks

        singleantlayout = [
            [
                sg.Column([
                    [sg.Listbox(listbox_items(tracks, info),
                                size=(20, 15), k=K.TrackList, enable_events=True,
                                select_mode=sg.LISTBOX_SELECT_MODE_BROWSE)]
                ]),
                sg.Column([
                    [MyGraph(info.video_shape, k=K.Graph2)],
                    [
                        sg.B("◀", k=K.FrameBack2),
                        sg.Slider(orientation='h', enable_events=True, k=K.FrameSlider2),
                        sg.B("▶", k=K.FrameForw2)
                    ],
                ])
            ]

        ]
        allvideolayout = [
            [sg.Column([
                [MyGraph(info.video_shape, k=K.Graph)],
                [
                    sg.B("◀", k=K.FrameBack),
                    sg.Slider(orientation='h', enable_events=True, k=K.FrameSlider),
                    sg.B("▶", k=K.FrameForw)
                ],
            ], expand_x=True,
            )]
        ]
        layout = [
            [sg.Checkbox("Filtrar", will_filter, enable_events=True, k=K.Filter),
             sg.Checkbox("Esconder máscara", enable_events=True, k=K.HideMask)],
            [sg.TabGroup(
                [[sg.Tab("Video completo", allvideolayout, k=K.Tab1),
                  sg.Tab("Por hormiga", singleantlayout, k=K.Tab2)]], k=K.Tabs,
                enable_events=True)]
        ]
        window = sg.Window(f"trkviz - {info.video_name}", layout, return_keyboard_events=True, finalize=True)
        g: MyGraph = window[K.Graph]  # noqa
        g2: MyGraph = window[K.Graph2]  # noqa
        window[K.FrameSlider].update(range=(0, info.video_length - 1))
        window[K.FrameSlider].update(value=0)
        window[K.FrameSlider].expand(expand_x=True)

        def empty_frame():
            return np.ones(info.video_shape + (3,), dtype='uint8') * 255

        def update_current_frame(curr_frame):
            image = video[curr_frame] if video else empty_frame()
            if not window[K.HideMask].get():
                image = Track.draw_tracks(info.tracks, image, curr_frame)
            g.draw_frame(image)

        def update_ant_pic(curr_frame):
            if selected_ant and selected_ant.at(curr_frame):
                image = video[curr_frame] if video else empty_frame()
                if not window[K.HideMask].get():
                    blob = selected_ant.at(curr_frame)
                    image = selected_ant.draw_blob(curr_frame, image).copy()
                    from skimage.draw import rectangle_perimeter, line

                    # draw bounding box
                    rect = blob.bbox
                    rr, cc = rectangle_perimeter(rect.topleft.yx, rect.bottomright.yx, shape=image.shape)
                    image[rr, cc] = Colors.GREEN

                    # draw leafdetector box
                    if blob.is_fully_visible(0.05):
                        from ..tracker.leafdetect import _get_blob_rect
                        rect = _get_blob_rect(
                            blob,
                            image.shape,
                            extra_pixels=15,
                            square=True
                        )
                        rr, cc = rectangle_perimeter(rect.topleft.yx, rect.bottomright.yx, shape=image.shape)
                        image[rr, cc] = Colors.RED
                        # image = draw_text(image, round(displayed_prob,2), rect.center).copy()

                    # draw center area
                    rect = Side.center_rect(image.shape, 0.05)
                    rr, cc = rectangle_perimeter(rect.topleft.yx, rect.bottomright.yx)
                    image[rr, cc] = Colors.BLUE

                    # draw length & width
                    props = blob.props
                    if props:
                        y0, x0 = int(props.centroid[0]), int(props.centroid[1])
                        orientation = props.orientation
                        x1 = int(x0 + math.cos(orientation) * 0.5 * props.minor_axis_length)
                        y1 = int(y0 - math.sin(orientation) * 0.5 * props.minor_axis_length)
                        x2 = int(x0 - math.sin(orientation) * 0.5 * props.major_axis_length)
                        y2 = int(y0 - math.cos(orientation) * 0.5 * props.major_axis_length)
                        x1 = clamp(x1, 0, info.video_shape[1]-1)
                        x2 = clamp(x2, 0, info.video_shape[1]-1)
                        y1 = clamp(y1, 0, info.video_shape[0]-1)
                        y2 = clamp(y2, 0, info.video_shape[0]-1)
                        rr, cc = line(y0, x0, y1, x1)
                        image[rr, cc] = blend(image[rr, cc], Colors.RED, 0.5)
                        rr, cc = line(y0, x0, y2, x2)
                        image[rr, cc] = blend(image[rr, cc], Colors.BLUE, 0.5)

                g2.draw_frame(image)

        selected_ant = None
        minframe = 0
        maxframe = info.video_length - 1
        update = update_current_frame
        slider = K.FrameSlider
        update_current_frame(0)
        # from ..tracker.leafdetect import TFLiteLeafDetector
        # detector = TFLiteLeafDetector(c.TFLITE_MODEL, video)
        # displayed_prob = 0
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == 'q':
                break
            if event == K.HideMask:
                update(int(values[slider]))
            if event == K.Filter:
                with LoadingWindow(spinner=C.SMALLSPINNER):
                    if window[K.Filter].get():
                        tracks = filtered_tracks
                    else:
                        tracks = info.tracks
                    window[K.TrackList].update(listbox_items(tracks, info), set_to_index=0)
            elif event == K.Tabs:
                window[K.Tab1].set_focus(False)
                window[K.Tab2].set_focus(False)
                if values[event] == K.Tab1:
                    minframe = 0
                    maxframe = info.video_length - 1
                    slider = K.FrameSlider
                    update = update_current_frame
                    window[slider].update(value=minframe, range=(minframe, maxframe))
                    update(0)
                else:
                    window[K.TrackList].update(set_to_index=0)
                    if len(info.tracks) > 0:
                        selected_ant = info.tracks[0]
                        minframe = selected_ant.first_frame()
                        maxframe = selected_ant.last_frame()
                        # displayed_prob = detector.probability(selected_ant)
                    else:
                        selected_ant = None;
                        minframe = maxframe = 0;
                    slider = K.FrameSlider2
                    window[slider].update(value=minframe, range=(minframe, maxframe))
                    update = update_ant_pic
                    update(minframe)
            elif event in (K.FrameSlider, K.FrameSlider2):
                update(int(values[event]))
            elif event in (K.FrameBack, K.FrameBack2, 'Left:37', 'MouseWheel:Up'):
                current_frame = int(values[slider])
                if current_frame > minframe:
                    window[slider].update(value=current_frame - 1)
                    update(current_frame - 1)
            elif event in (K.FrameForw, K.FrameForw2, 'Right:39', 'MouseWheel:Down'):
                current_frame = int(values[slider])
                if current_frame < maxframe:
                    window[slider].update(value=current_frame + 1)
                    update(current_frame + 1)
            elif event == K.TrackList:
                if len(values[event]):
                    i = pull_track_id(values[event][0])
                    track = Track.get(info.tracks, i)
                    selected_ant = track
                    minframe = selected_ant.first_frame()
                    maxframe = selected_ant.last_frame()
                    window[slider].update(value=minframe, range=(minframe, maxframe))
                    # displayed_prob = detector.probability(selected_ant)
                    update_ant_pic(minframe)
    finally:
        LoadingWindow.close_all()

if __name__ == '__main__':
    freeze_support()
    trkviz()
