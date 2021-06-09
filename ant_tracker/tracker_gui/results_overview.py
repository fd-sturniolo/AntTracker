import PySimpleGUI as sg
import os
import pims
import threading
from pathlib import Path
from pims.process import crop

from . import constants as C
from .export import Exporter
from .extracted_parameters import SelectionStep
from .guicommon import align, change_bar_color, write_event_value_closure
from .parameter_extraction import extract_pixel_size
from .session import SessionInfo
from ..tracker.common import Rect, crop_from_rect
from ..tracker.info import TracksCompleteInfo, Direction
from ..tracker.leafdetect import TFLiteLeafDetector
from ..tracker.segmenter import LogWSegmenter
from ..tracker.track import Track
from ..tracker.tracking import Tracker

PB_DEEP_BLUE = "#082567"
PB_GREEN = "#01826B"
PB_GRAY = '#D0D0D0'

class K:
    Scroll = '-SCROLL-'
    Cancel = '-CANCEL-'
    Export = '-EXPORT-'
    OpenExport = '-OPEN_EXPORT-'
    Report = '-REPORT-'
    ThreadFinished = '-THREAD_DONE-'

PBAR_HEIGHT = 10
PBAR_HEIGHT_PAD = 11

def short_fn(path):
    return "/".join(path.parts[-2:])
def filelabel(path: Path):
    return f'-FLABEL!!{short_fn(path)}-'
def progbar(path: Path):
    return f'-BAR!!{short_fn(path)}-'
def is_progbar(key: str):
    return key.startswith('-BAR!!')
def finished(path: Path):
    return f'-FINISHED!!{short_fn(path)}-'
def is_finished(key: str):
    return key.startswith('-FINISHED!!')
def get_finished_btn(key: str):
    return '-LABEL!!' + key.split('!!')[1]
def antlabel(path: Path):
    return f'-LABEL!!{short_fn(path)}-'
def is_antlabel(key: str):
    return key.startswith('-LABEL!!')

def open_labeler(trkfile):
    exp_layout = [
        [sg.Text("AntLabeler aún no permite revisar tracks.")],
        [sg.Text(
            "Sin embargo, existe un visor experimental para ver el resultado del tracking sobre el video.")],
        [sg.Text("Desea utilizarlo?")],
        [sg.Ok("Sí"), sg.Cancel("No")],
    ]
    resp = sg.Window("No disponible", exp_layout,
                     icon=C.LOGO_AT_ICO, modal=True, keep_on_top=True).read(close=True)[0]
    if resp == 'Sí':
        from .trkviz import trkviz_subprocess
        trkviz_subprocess(trkfile)

def filter_func(info: TracksCompleteInfo):
    f = info.filter_func(**C.TRACKFILTER)
    return lambda t: info.track_direction(t) != Direction.UN and f(t)

class TrackTask:
    def __init__(self, save_every_n_leaf_detects=100):
        self._running = True
        self.exc = None
        self.save_every_n_leaf_detects = save_every_n_leaf_detects

    def terminate(self):
        self._running = False

    def run(self, window: sg.Window, sesspath: Path, with_leaves: bool):
        try:
            self.inner_run(window, sesspath, with_leaves)
        except:  # noqa
            import sys
            self.exc = sys.exc_info()

    def inner_run(self, window: sg.Window, sesspath: Path, with_leaves: bool):
        send = write_event_value_closure(window)
        session = SessionInfo.load(sesspath, with_trackers=True)
        start_time = session.first_start_time
        for path in session.videofiles:
            p = session.parameters[path]
            crop_rect: Rect = p.rect_data[SelectionStep.TrackingArea]
            marker: Rect = p.rect_data[SelectionStep.SizeMarker]
            pixel_size_in_mm = extract_pixel_size(marker)
            trkfile = session.get_trkfile(path)

            progress_key = progbar(path)
            S = SessionInfo.State

            if session.states[path] == S.GotParameters:
                send(K.Report, f"{short_fn(path)}, comenzando tracking...")
            if session.states[path] == S.Tracking:
                send(K.Report, f"{short_fn(path)}, recuperando tracking...")
            if session.states[path] == S.DetectingLeaves:
                send(K.Report, f"{short_fn(path)}, comenzando detección de hojas...")
            video = None
            info = None

            if session.states[path] < S.Finished:
                send(K.Report, f"{short_fn(path)}, cargando video...")
                video = pims.PyAVReaderIndexed(path)
                video = crop(video, crop_from_rect(video.frame_shape[0:2], crop_rect))
            if not info and S.DetectingLeaves <= session.states[path] <= S.Finished:
                send(K.Report, f"{short_fn(path)}, cargando información previa...")
                info = TracksCompleteInfo.load(trkfile)

            if session.states[path] in (S.GotParameters, S.Tracking):
                if session.states[path] == S.GotParameters:
                    tracker = Tracker(path, LogWSegmenter(video, p.segmenter_parameters), params=p.tracker_parameters)
                    track_generator = tracker.track_progressive()
                else:  # if session.states[path] == S.Tracking:
                    tracker = session.unfinished_trackers[path]
                    send(progress_key, {'p': tracker.last_tracked_frame})
                    send(K.Report, f"{short_fn(path)}, recuperando tracking desde: "
                                   f"{tracker.last_tracked_frame}/{tracker.video_length}")
                    track_generator = tracker.track_progressive_continue()

                for frame in track_generator:
                    send(K.Report, f"{short_fn(path)}, tracking: {frame}/{tracker.video_length}")
                    send(progress_key, {'p': frame})
                    if session.save_every_n_frames > 0 and frame > 0 and frame % session.save_every_n_frames == 0:
                        session.states[path] = S.Tracking
                        send(K.Report, f"{short_fn(path)}, guardando tracking hasta frame {frame}...")
                        session.record_tracker_state(path, tracker)
                        session.save(sesspath)
                    if not self._running:
                        return
                send(K.Report, f"{short_fn(path)}, guardando tracking...")
                info = TracksCompleteInfo(tracker.info(), pixel_size_in_mm, crop_rect, p.nest_side,
                                          start_time=start_time)
                info.save(trkfile)
                session.states[path] = S.DetectingLeaves
                session.save(sesspath)
            send(progress_key, {'color': PB_GREEN, 'background': PB_DEEP_BLUE})

            # sólo detectar hojas para tracks que van al export
            _load_relevant = filter_func(info)
            load_relevant_tracks = [track for track in info.tracks if _load_relevant(track)]
            track_ids_still_undetected = [track.id for track in load_relevant_tracks if not track.load_detected]
            if with_leaves and session.states[path] == S.Finished:
                if track_ids_still_undetected:
                    session.states[path] = S.DetectingLeaves
                    send(K.Report, f"{short_fn(path)}, cargando video...")
                    video = pims.PyAVReaderIndexed(path)
                    video = crop(video, crop_from_rect(video.frame_shape[0:2], crop_rect))

            if session.states[path] == S.DetectingLeaves:
                if with_leaves:
                    send(K.Report, f"{short_fn(path)}, comenzando detección de hojas...")
                    tracks = load_relevant_tracks
                    send(progress_key, {'p': 1, 'max': len(tracks) + 1})
                    detected_ids = session.detection_probs[path].keys()
                    to_detect = set(track_ids_still_undetected) - detected_ids
                    send(progress_key, {'p': len(tracks) - len(to_detect), 'max': len(tracks) + 1})
                    detector = TFLiteLeafDetector(C.TFLITE_MODEL, video)
                    send(K.Report, f"{short_fn(path)}, retomando detección de hojas... ")
                    for i_trk, track in enumerate(tracks):
                        if track.id in to_detect:
                            send(K.Report, f"{short_fn(path)}, detectando hojas: {i_trk}/{len(tracks)}")
                            session.record_detection(path, track, detector.probability(track))
                            if i_trk > 0 and i_trk % self.save_every_n_leaf_detects == 0:
                                send(K.Report, f"{short_fn(path)}, guardando detecciones hasta hormiga {i_trk}...")
                                session.save(sesspath)
                            send(progress_key, {'p': i_trk + 1})
                        if not self._running:
                            return
                    send(K.Report, f"{short_fn(path)}, guardando detecciones de hoja...")
                    for track_id, prob in session.detection_probs[path].items():
                        Track.get(info.tracks, track_id).set_load_probability(prob)
                info = TracksCompleteInfo(info, pixel_size_in_mm, crop_rect, p.nest_side, start_time=start_time)
                info.save(trkfile)
                session.states[path] = S.Finished
                session.save(sesspath)
            if session.states[path] == S.Finished:
                send(K.Report, f"{short_fn(path)}, finalizado.")
                send(progress_key, {'p': 1, 'max': 1, 'visible': False})
                send(finished(path), {'trkfile': trkfile})
                start_time = info.end_time
        send(K.ThreadFinished, None)

def results_overview(sesspath: Path, with_leaves=True):
    from .loading_window import LoadingWindow
    with LoadingWindow("Configurando..."):
        session = SessionInfo.load(sesspath)
        exportpath = sesspath.parent / "export.xlsx"
        paths = session.videofiles
        bars_column_height = 250
        win_height = bars_column_height + 100
        win_width = 600
        fn_width = int(max([len(short_fn(p)) for p in paths]) * 0.9)
        inner_width = win_width - 80

        layout = [
            [sg.Column([
                [sg.ProgressBar(10, 'h', (0, 0), pad=(0, 0), k='-SPACER_BAR-')],
                *[[sg.T(short_fn(path).rjust(fn_width), size=(fn_width, 1), justification='right',
                        k=filelabel(path)),
                   sg.ProgressBar(session.lengths[path], 'h', (0, PBAR_HEIGHT),
                                  bar_color=(PB_DEEP_BLUE, PB_GRAY),
                                  pad=(0, PBAR_HEIGHT_PAD), k=progbar(path)),
                   sg.Button("Revisar etiquetado", visible=False, k=antlabel(path))] for path in paths]],

                scrollable=True, vertical_scroll_only=True, expand_x=True, size=(win_width, bars_column_height),
                k=K.Scroll,
            )],
            [sg.HorizontalSeparator()],
            [sg.InputText("Recuperando datos...", k=K.Report, disabled=True, visible=True)],
            [align([sg.vbottom([
                sg.pin(sg.Button("Abrir archivo", k=K.OpenExport, visible=False)),
                sg.pin(sg.Button("Exportar resultados", k=K.Export, visible=False)),
                sg.Button("Detener", k=K.Cancel)]
            )], 'right')],
        ]
        window = sg.Window("AntTracker - Procesando...", layout, icon=C.LOGO_AT_ICO,
                           size=(win_width, win_height), disable_close=True, finalize=True)
        for path in paths:
            lw, _ = window[filelabel(path)].get_size()
            window[progbar(path)].Widget.configure(length=inner_width - lw)
        window['-SPACER_BAR-'].Widget.configure(length=inner_width)
        window[K.Report].expand(expand_x=True)
        x, y = window.CurrentLocation()
        window.move(x, y - 100)

        task = TrackTask()
        t = threading.Thread(target=task.run, args=(window, sesspath, with_leaves), daemon=True)
        t.start()

    while True:
        event, values = window.read(1000)
        if not t.is_alive() and task.exc:  # errored out
            raise task.exc[0](f"TaskThread threw an exception: {task.exc[1]}").with_traceback(task.exc[2])
        if event == sg.TIMEOUT_EVENT:
            pass
        elif event == sg.WIN_CLOSED or event == K.Cancel:
            msg = ("¿Está seguro de que desea detener el procesamiento? "
                   "Los videos ya procesados mantendrán sus resultados.\n"
                   "El progreso en el video actual es guardado períodicamente y se restaurará cuando continue.")
            if ((not t.is_alive()) or
                    sg.popup(msg, modal=True, custom_text=(C.RESP_SI, C.RESP_NO)) == C.RESP_SI):
                task.terminate()
                window.close()
                del window
                return None
        elif is_progbar(event):
            val = values[event]
            pbar: sg.ProgressBar = window[event]  # noqa
            if 'color' in val:
                change_bar_color(pbar, val['color'], val.get('background', None))
            if 'p' in val:
                pbar.update(val['p'], max=val.get('max', None), visible=val.get('visible', None))
        elif event == K.Report:
            window[event].update(values[event])
        elif is_finished(event):
            btn_key = get_finished_btn(event)
            window[btn_key].update(visible=True)
            window[btn_key].expand(expand_x=True)
            window[btn_key].metadata = values[event]
        elif event == K.ThreadFinished:
            window[K.Cancel].update("Cerrar")
            window[K.Export].update(visible=True)
            window.set_title("AntTracker - Resultados")
        elif is_antlabel(event):
            trkfile = window[event].metadata['trkfile']
            open_labeler(trkfile)
        elif event == K.OpenExport:
            with LoadingWindow():
                os.startfile(exportpath)
                window[K.OpenExport].update("Abriendo...", disabled=True)

                def wait_n_update():
                    from time import sleep
                    sleep(10)
                    window[K.OpenExport].update("Abrir archivo", disabled=False)

                threading.Thread(target=wait_n_update, daemon=True).start()
        elif event == K.Export:
            session = SessionInfo.load(sesspath)
            trkfiles = [session.get_trkfile(f) for f in session.videofiles]
            infos = []
            title = "Exportando..."
            sg.OneLineProgressMeter(title, 0, len(trkfiles), 'XP', "Cargando tracks...")
            for i_f, f in enumerate(trkfiles, 1):
                infos.append(TracksCompleteInfo.load(f))
                sg.OneLineProgressMeter(title, i_f, len(trkfiles), 'XP', "Cargando tracks...")
            ex = Exporter()
            for msg, progress, progmax in ex.export_progress(infos):
                sg.OneLineProgressMeter(title, progress, progmax, 'XP', msg)
            while True:
                try:
                    ex.save(exportpath)
                    break
                except PermissionError:
                    sg.popup_ok(f"El siguiente archivo está abierto o protegido:",
                                exportpath,
                                f"Presione Ok para probar nuevamente.", title="Error", line_width=len(str(exportpath)))
            window[K.Report].update(f"Exportado a: {exportpath}")
            window[K.OpenExport].update(visible=True)
        else:
            print(event)
