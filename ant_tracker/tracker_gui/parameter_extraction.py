from math import sqrt
import PySimpleGUI as sg
import numpy as np
from itertools import chain
from pathlib import Path
from typing import Dict, List, Union, Optional

from . import constants as C
from .extracted_parameters import SelectionStep, ExtractedParameters
from .guicommon import parse_number, release
from ..tracker.blob import Blob
from ..tracker.common import Side, Rect, Video, ensure_path
from ..tracker.parameters import SegmenterParameters, TrackerParameters, LogWSegmenterParameters
from ..tracker.segmenter import LogWSegmenter

class MyGraph(sg.Graph):
    def __init__(self, imshape, *args, **kwargs):
        import screeninfo

        screen_height = min([monitor.height for monitor in screeninfo.get_monitors()])
        screen_width = min([monitor.width for monitor in screeninfo.get_monitors()])
        video_height = imshape[0]
        video_width = imshape[1]

        ideal_video_height = screen_height * 0.70
        ideal_video_width = screen_width * 0.65

        height_scale = ideal_video_height / video_height
        width_scale = ideal_video_width / video_width

        # If scaling width to ideal makes the height go over, scale by height instead
        if video_height*width_scale > ideal_video_height:
            self.video_scale = height_scale
        # Otherwise, do scale by width
        else:
            self.video_scale = width_scale

        self.default_line_width = np.ceil(1 / self.video_scale)
        graph_shape = int(video_width * self.video_scale), int(video_height * self.video_scale)
        self.frame_id = None

        super(MyGraph, self).__init__(graph_shape, (0, video_height - 1), (video_width - 1, 0), *args, **kwargs)

    def draw_frame(self, frame: np.ndarray):
        from PIL import Image
        import io

        if self.frame_id is not None:
            self.delete_figure(self.frame_id)
        buf = io.BytesIO()
        Image.fromarray(frame).resize(self.CanvasSize).save(buf, format='PNG')
        self.frame_id = self.draw_image(data=buf.getvalue(), location=(0, 0))
        self.TKCanvas.tag_lower(self.frame_id)

    def DrawRectangle(self, rect: Rect,
                      fill_color: str = None, line_color: str = None, line_width: int = None, **kwargs
                      ) -> Union[int, None]:
        """
        Draw a rectangle given a `Rect`. Can control the line and fill colors,
        and also pass kwargs to underlying tk call.
        :param rect: the rectangle to draw
        :param fill_color: color of the interior
        :param line_color: color of outline
        :param line_width: width of the line in pixels
        :return: id returned from tkinter that you'll need if you want to manipulate the rectangle
        """

        converted_top_left = self._convert_xy_to_canvas_xy(rect.topleft.x, rect.topleft.y)
        converted_bottom_right = self._convert_xy_to_canvas_xy(rect.bottomright.x, rect.bottomright.y)
        if self._TKCanvas2 is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        if line_width is None:
            line_width = self.default_line_width
        try:  # in case closed with X
            _id = self._TKCanvas2.create_rectangle(converted_top_left[0], converted_top_left[1],
                                                   converted_bottom_right[0],
                                                   converted_bottom_right[1], fill=fill_color, outline=line_color,
                                                   width=line_width, **kwargs)
        except:  # noqa
            _id = None
        return _id

    draw_rectangle = DrawRectangle

class K:
    Graph = '-GRAPH-'
    FrameSlider = '-FRAME_SLIDER-'
    FrameBack = '-FRAME_BACK-'
    FrameForw = '-FRAME_FORWARD-'
    SelectInstructions = '-SELECT_INSTRUCTIONS-'
    BackButton = '-BACK_BUTTON-'
    ContinueButton = '-START_TRACK_BUTTON-'
    ShowDetailsButton = '-SHOW_PARAM_DETS-'
    ParameterDetails = '-PARAM_DETS-'
    PreviewButton = '-PREVIEW_BUTTON-'
    RefreshButton = '-REFRESH_BUTTON-'

def extract_parameters_from_video(video: Video, filepath: Union[Path, str]):
    filepath = ensure_path(filepath)
    imshape = video[0].shape[0:2]

    _segmenterParameters = LogWSegmenterParameters()
    _trackerParameters = TrackerParameters(use_defaults=True)
    parameters_gotten = False

    def wrap(s):
        import textwrap
        return "\n".join(textwrap.wrap(s, 70))

    details_list = sg.Column(
        [
            *[[sg.Text(n, size=(28, 1), pad=(0, 0), tooltip=wrap(d)),
               sg.Text("❔", font=("Arial", 8), size=(2, 1), tooltip=wrap(d), pad=(0, 0)),
               sg.InputText(v, size=(7, 1), pad=(0, 0), k=k)] for k, n, d, v in
              chain(
                  zip(_segmenterParameters.keys(),
                      _segmenterParameters.names(),
                      _segmenterParameters.descriptions(),
                      _segmenterParameters.values()),
                  zip(_trackerParameters.keys(),
                      _trackerParameters.names(),
                      _trackerParameters.descriptions(),
                      _trackerParameters.values())
              )
              ],
            [sg.B("Actualizar", k=K.RefreshButton)]],
        element_justification='right', visible=False, k=K.ParameterDetails)

    def update_params(win, segParams: SegmenterParameters, trackerParams: TrackerParameters):
        [win[k](v) for k, v in chain(segParams.items(), trackerParams.items())]

    def get_params(win):
        return (SegmenterParameters({k: parse_number(win[k].get()) for k, _ in _segmenterParameters.items()}),
                TrackerParameters({k: parse_number(win[k].get()) for k, _ in _trackerParameters.items()}))

    layout = [
        [
            sg.Column(
                [
                    [sg.Text("1. Encierre en un rectángulo la posición del marcador de 10mm.", k=K.SelectInstructions)],
                    [
                        sg.pin(sg.B("Retroceder", visible=False, k=K.BackButton)),
                        sg.pin(sg.B("Mostrar parámetros", visible=False, k=K.ShowDetailsButton)),
                        sg.pin(sg.B("Previsualizar", visible=False, k=K.PreviewButton)),
                        sg.pin(sg.B("Continuar", visible=False, k=K.ContinueButton)),
                    ],
                    [details_list]
                ],
                vertical_alignment='top', expand_y=True, expand_x=True
            ),
            sg.Column(
                [[MyGraph(imshape, k=K.Graph, enable_events=True, drag_submits=True)],
                 [
                     sg.B("◀", k=K.FrameBack),
                     sg.Slider(orientation='h', enable_events=True, k=K.FrameSlider),
                     sg.B("▶", k=K.FrameForw)],
                 ], expand_x=True,
            )
        ],
        [sg.Button('Go', visible=False), sg.Button('Nothing', visible=False), sg.Button('Exit', visible=False)]
    ]

    window = sg.Window(f"AntTracker - {filepath}", layout, icon=C.LOGO_AT_ICO, modal=True, finalize=True)
    window[K.SelectInstructions].expand(expand_y=True, expand_x=True)
    g: MyGraph = window[K.Graph]  # noqa

    g.draw_frame(video[0])
    previewing_segmentation = False

    window[K.FrameSlider].update(range=(0, len(video) - 1))
    window[K.FrameSlider].update(value=0)
    window[K.FrameSlider].expand(expand_x=True)

    rect_start = (0, 0)

    dragging = False

    rect: Rect = None  # noqa
    selection_id = None
    rect_colors = {SelectionStep.SizeMarker: "tomato", SelectionStep.TrackingArea: "SlateBlue1",
                   SelectionStep.AntFrame1:  "DodgerBlue2", SelectionStep.AntFrame2: "DodgerBlue4"}
    rect_names = {SelectionStep.SizeMarker: "Marcador", SelectionStep.TrackingArea: "Área de tracking",
                  SelectionStep.AntFrame1:  "H1", SelectionStep.AntFrame2: "H2"}
    rect_ids: Dict[SelectionStep, List[int]] = {step: [] for step in rect_names.keys()}
    rect_data: Dict[SelectionStep, Optional[Rect]] = {step: None for step in rect_names.keys()}

    def calc_parameters(r_data: Dict[SelectionStep, Rect]):
        _sp = LogWSegmenterParameters()
        _tp = TrackerParameters(use_defaults=True)

        antrect1 = r_data[SelectionStep.AntFrame1]
        antrect2 = r_data[SelectionStep.AntFrame2]
        average_ant_diagonal = (antrect1.diagonal_length + antrect2.diagonal_length) / 2

        _sp.approx_tolerance = max(1,round(0.55 * sqrt(average_ant_diagonal) - 0.87, 2))
        _sp.gaussian_sigma = round(0.08 * average_ant_diagonal + 5, 2)
        _sp.minimum_ant_radius = round(0.1 * average_ant_diagonal)

        _tp.max_distance_between_assignments = round(antrect1.center.distance_to(antrect2.center) * 2, 2)

        return _sp, _tp

    def _w(x):
        return int(x * .08)

    def _w_1(x):
        return int(x * (1 - .08))

    nest_sides: Dict[Side, Rect] = {
        Side.Top:    Rect.from_points((0, 0), (imshape[1] - 1, _w(imshape[0]))),
        Side.Left:   Rect.from_points((0, 0), (_w(imshape[1]), imshape[0] - 1)),
        Side.Right:  Rect.from_points((_w_1(imshape[1]), 0), (imshape[1] - 1, imshape[0] - 1)),
        Side.Bottom: Rect.from_points((0, _w_1(imshape[0])), (imshape[1] - 1, imshape[0] - 1)),
    }
    nest_side_ids = {}
    side_rect = None
    hover_rect = None
    chosen_side = None
    current_step = SelectionStep.First

    def set_selection_instructions(win, step):
        def selection_instructions(_step):
            ret = "1. Encierre en un rectángulo la posición del marcador de 10mm."
            if _step == SelectionStep.SizeMarker: return ret
            ret += "\n2. Encierre el área donde se realizará el tracking. Considere que mientras mayor sea el área, " \
                   "mayor es el tiempo de procesamiento necesario, pero más precisas serán las medidas obtenidas " \
                   "a lo largo del tiempo."
            if _step == SelectionStep.TrackingArea: return ret
            ret += "\n3. Encierre la totalidad del tamaño de una hormiga promedio en movimiento. En lo posible, " \
                   "busque una que se mueva en línea recta."
            if _step == SelectionStep.AntFrame1: return ret
            ret += "\n4. Avance 5 cuadros, y encierre la nueva posición de la misma hormiga."
            if _step == SelectionStep.AntFrame2: return ret
            ret += "\n5. Haga click en el lado que corresponde a la ubicación del nido."
            if _step == SelectionStep.NestSide: return ret
            ret += "\n6. Puede revisar los parámetros en el menú que se encuentra debajo," \
                   "y previsualizar la segmentación con el botón correspondiente. " \
                   "Si está de acuerdo, presione Continuar."
            return ret

        win[K.SelectInstructions].update(value=selection_instructions(step))

    def update_current_frame(frame):
        image = video[frame]
        if previewing_segmentation and frame > 0:
            progbar = sg.ProgressBar(2, orientation='h', size=(20, 20))
            progwin = sg.Window('Previsualizando', [[progbar]], modal=True, disable_close=True).finalize()
            params = get_params(window)[0]
            n = min(params.movement_detection_history, frame)
            progbar.update_bar(0)
            blobs = LogWSegmenter.segment_single(params, image, video[frame - n:frame])
            progbar.update_bar(1)
            image = Blob.draw_blobs(blobs, image)
            progbar.update_bar(2)
            progwin.close()
        g.draw_frame(image)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            window.close()
            return None
        elif event == K.FrameSlider:
            update_current_frame(int(values[K.FrameSlider]))
        elif event == K.FrameBack:
            current_frame = int(values[K.FrameSlider])
            if current_frame > 0:
                window[K.FrameSlider].update(value=current_frame - 1)
                update_current_frame(current_frame - 1)
        elif event == K.FrameForw:
            current_frame = int(values[K.FrameSlider])
            if current_frame < len(video):
                window[K.FrameSlider].update(value=current_frame + 1)
                update_current_frame(current_frame + 1)
        elif event == K.Graph:
            if current_step != SelectionStep.Done:
                if current_step == SelectionStep.NestSide:
                    for side, rect in nest_sides.items():
                        if side_rect is None and values[K.Graph] in rect:
                            chosen_side = side
                            side_rect = g.draw_rectangle(rect, line_width=0, line_color="green", fill_color="green",
                                                         stipple="gray50")
                            current_step = current_step.next()
                            break
                elif not dragging:
                    dragging = True
                    rect_start = values[K.Graph]
                elif dragging:
                    if selection_id is not None:
                        g.delete_figure(selection_id)
                    rect = Rect.from_points(rect_start, values[K.Graph]).clip(imshape)
                    selection_id = g.draw_rectangle(rect, line_color=rect_colors[current_step])
        elif event == release(K.Graph) and dragging:
            if current_step != SelectionStep.Done:
                rect_ids[current_step].append(
                    g.draw_rectangle(rect, line_color=rect_colors[current_step])
                )
                rect_ids[current_step].append(
                    g.draw_text(rect_names[current_step], rect.topleft, rect_colors[current_step],
                                text_location=sg.TEXT_LOCATION_BOTTOM_LEFT)
                )
                rect_data[current_step] = rect
                if current_step == SelectionStep.AntFrame2:
                    start = rect_data[SelectionStep.AntFrame1]
                    end = rect_data[SelectionStep.AntFrame2]
                    rect_ids[current_step].append(
                        g.draw_line(start.center, end.center, color=rect_colors[current_step])
                    )

                current_step = current_step.next()
            if selection_id is not None:
                g.delete_figure(selection_id)
            dragging = False
        elif event == K.BackButton:
            window[K.ParameterDetails].Visible = False
            previewing_segmentation = False
            current_step = current_step.back()
            if current_step in rect_ids:
                [g.delete_figure(i) for i in rect_ids[current_step]]
                rect_ids[current_step] = []
                rect_data[current_step] = None
        elif event == K.ShowDetailsButton:
            window[K.ParameterDetails].Visible = not window[K.ParameterDetails].Visible
        elif event == K.PreviewButton:
            previewing_segmentation = not previewing_segmentation
            update_current_frame(int(values[K.FrameSlider]))
        elif event == K.RefreshButton:
            update_current_frame(int(values[K.FrameSlider]))
        elif event == K.ContinueButton:
            sp, tp = get_params(window)
            window.close()
            return ExtractedParameters(sp, tp, rect_data, chosen_side)
        else:
            print(event, values)

        if current_step == SelectionStep.NestSide:
            if not nest_side_ids:
                def draw_hover_rect(r: Rect):
                    def _(_):
                        nonlocal hover_rect
                        hover_rect = g.draw_rectangle(r, line_color="green", fill_color="green",
                                                      stipple="gray50",
                                                      line_width=0)
                        g.TKCanvas.tag_bind(hover_rect, '<Leave>', delete_hover_rect)

                    return _

                def delete_hover_rect(_):
                    nonlocal hover_rect
                    if hover_rect is not None:
                        g.delete_figure(hover_rect)  # noqa
                    hover_rect = None

                nest_side_ids = {
                    side: g.draw_rectangle(r, line_color="green", fill_color="green", stipple="gray12",
                                           line_width=0)
                    for side, r in
                    nest_sides.items()}
                for side, i in nest_side_ids.items():
                    g.TKCanvas.tag_bind(i, '<Enter>', draw_hover_rect(nest_sides[side]))
        elif nest_side_ids:
            [g.delete_figure(i) for i in nest_side_ids.values()]
            nest_side_ids = {}
        if current_step <= SelectionStep.NestSide:
            if side_rect is not None: g.delete_figure(side_rect)
            side_rect = None
        if current_step == SelectionStep.Done:
            if not parameters_gotten:
                sp, tp = calc_parameters(rect_data)
                update_params(window, sp, tp)
                parameters_gotten = True
        else:
            parameters_gotten = False

        set_selection_instructions(window, current_step)
        window[K.BackButton].update(visible=current_step != SelectionStep.First)
        window[K.ShowDetailsButton].update(visible=current_step == SelectionStep.Done)
        window[K.PreviewButton].update(visible=current_step == SelectionStep.Done)
        window[K.ContinueButton].update(visible=current_step == SelectionStep.Done)
        window[K.ParameterDetails].update(visible=window[K.ParameterDetails].Visible)

def extract_pixel_size(marker: Rect):
    diameter_px = (marker.height + marker.width) / 2
    pixel_size_in_mm = 10 / diameter_px
    return pixel_size_in_mm
