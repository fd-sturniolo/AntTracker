import PySimpleGUI as sg
import av
import datetime
import pims
from itertools import chain
from pathlib import Path
from typing import Union, List

from . import constants as C
from .loading_window import LoadingWindow
from .parameter_extraction import extract_parameters_from_video
from .results_overview import results_overview
from .session import SessionInfo

def get_datetime():
    sg.theme('BlueMono')
    win = sg.Window("Ingrese fecha y hora del primer video",
                    [[sg.T("Fecha:"), sg.In("", disabled=True, enable_events=True, k='-DATE_INP-'),
                      sg.CalendarButton("Abrir calendario", target='-DATE_INP-', title="Elegir fecha",
                                        format="%Y-%m-%d",
                                        enable_events=True, k='-DATE-')],
                     [sg.T("Hora:"), sg.In("HH:MM:SS", disabled=True, k='-TIME_INP-')],
                     [sg.B('OK', disabled=True), sg.B('Cancelar')]], finalize=True, return_keyboard_events=True)
    while True:
        event, values = win.read()
        if event in (sg.WIN_CLOSED, 'Cancelar'):
            win.close()
            sg.theme('Default1')
            return None
        elif event == 'OK':
            win.close()
            sg.theme('Default1')
            return dtime  # noqa
        date = win['-DATE_INP-'].get()
        win['-TIME_INP-'].update(disabled=(date == ""))
        time = win['-TIME_INP-'].get()
        try:
            dtime = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
            win['OK'].update(disabled=False)
        except ValueError:
            win['OK'].update(disabled=True)

def load_video(path: Union[Path, str], message="Cargando video..."):
    with LoadingWindow(message, C.SMALLSPINNER):
        v = pims.PyAVReaderIndexed(path)
    return v
def get_video_len(path: Union[Path, str], message="Cargando video..."):
    with LoadingWindow(message, C.SMALLSPINNER):
        vlen = av.open(str(path)).streams.video[0].frames  # noqa
    return vlen

def ant_tracker_routine():
    folder = sg.popup_get_folder("Seleccione la carpeta con los videos a procesar", no_window=True)
    if not folder: return
    folder = Path(folder)

    from .excepthook import make_excepthook
    import sys
    sys.excepthook = make_excepthook(Path(folder), True)

    session: SessionInfo
    sesspath = Path(folder) / ".anttrackersession"

    files = list(chain(folder.glob("*.mp4"), folder.glob("*.avi"), folder.glob("*.h264")))

    def check_frame_info(_files):
        with LoadingWindow("Validando videos..."):
            import av
            valid_files: List[Path] = []
            invalid_files: List[Path] = []
            for f in _files:
                container = av.open(str(f))  # noqa
                if container.streams.video[0].frames != 0:
                    valid_files.append(f)
                else:
                    invalid_files.append(f)

        if invalid_files:
            invmsg = \
                [
                    "Se encontraron archivos de video con información faltante dentro de la carpeta. "
                    "Normalmente esto ocurre con videos con extensión .h264 obtenidos en bruto de AntVRecord. "
                    "Deberá obtener los videos procesados luego de finalizada la grabación "
                    "(generalmente con extensión .mp4).",
                    "Los videos inválidos son:",
                    "",
                ] + ["/".join(f.parts[-2:]) for f in invalid_files] + [
                    "",
                    "Los mismos serán ignorados al procesar."
                ]
            sg.PopupOK("\n".join(invmsg), title="Información faltante", modal=True, icon=C.LOGO_AT_ICO)
        return valid_files

    if sesspath.exists():
        msg = ("Parece ser que el proceso de tracking en esta carpeta ya fue comenzado.\n"
               "¿Desea continuar desde el último punto?\n\n"
               "⚠ ¡Cuidado! Si elije \"No\", se perderá todo el"
               " progreso previo y el procesamiento comenzará de cero.")
        response = sg.Popup(msg, title="Sesión previa detectada", custom_text=(C.RESP_SI, C.RESP_NO),
                            modal=True, icon=C.LOGO_AT_ICO)
        if response == C.RESP_SI:
            with LoadingWindow("Cargando sesión previa..."):
                session = SessionInfo.load(sesspath)
                newfiles = [f for f in files if f not in session.videofiles]
                deletedfiles = [f for f in session.videofiles if f not in files]
            newfiles = check_frame_info(newfiles)
            if newfiles or deletedfiles:
                with LoadingWindow("Cargando sesión previa..."):
                    session.add_new_files(newfiles)
                    session.remove_deleted_files(deletedfiles)
                    session.save(sesspath)
        elif response == C.RESP_NO:
            start_date = get_datetime()
            if start_date is None: return
            files = check_frame_info(files)
            session = SessionInfo.first_run(files, start_date)
        else:
            return
    else:
        start_date = get_datetime()
        if start_date is None: return
        files = check_frame_info(files)
        session = SessionInfo.first_run(files, start_date)

    # noinspection PyUnboundLocalVariable
    files = session.videofiles

    if len(files) == 0:
        sg.PopupOK("No se encontraron videos válidos en esta carpeta.", title="No hay videos", icon=C.LOGO_AT_ICO)
        return

    if session.is_first_run:
        video = load_video(files[0])[:]

        p = extract_parameters_from_video(video, files[0])
        if p is None: return
        session.parameters[files[0]] = p
        session.states[files[0]] = SessionInfo.State.GotParameters
        session.lengths[files[0]] = len(video)

        single_video = (len(files) == 1) or (
                sg.Popup("¿Desea usar los parámetros determinados en este video para el resto"
                         " de los videos del lote?\nSi decide no hacerlo, deberá realizar"
                         " este proceso para cada uno de los videos.",
                         title="Continuar", custom_text=(C.RESP_SI, C.RESP_NO),
                         modal=True) == C.RESP_SI)

        if single_video:
            session.parameters = {f: p for f in files}
            session.states = {f: SessionInfo.State.GotParameters for f in files}
            for i_file, file in enumerate(files[1:], 2):
                session.lengths[file] = get_video_len(file, f"Cargando video {i_file} de {len(files)}")
            session.save(sesspath)
        else:
            # noinspection DuplicatedCode
            for i_file, file in enumerate(files[1:], 2):
                video = load_video(file, f"Cargando video {i_file} de {len(files)}")[:]
                p = extract_parameters_from_video(video, file)
                if p is None: return
                session.parameters[file] = p
                session.states[file] = SessionInfo.State.GotParameters
                session.lengths[file] = len(video)
                session.save(sesspath)

    else:  # not the first run
        new_files = [p for p, s in session.states.items() if s == SessionInfo.State.New]
        # noinspection DuplicatedCode
        for i_file, file in enumerate(new_files, 1):
            video = load_video(file, f"Cargando video {i_file} de {len(new_files)}")[:]
            p = extract_parameters_from_video(video, file)
            if p is None: return
            session.parameters[file] = p
            session.states[file] = SessionInfo.State.GotParameters
            session.lengths[file] = len(video)
            session.save(sesspath)

    results_overview(sesspath)
