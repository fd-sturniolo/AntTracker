import PySimpleGUI as sg
import sys
from pathlib import Path

from . import constants as C
from .about import about
from .excepthook import make_excepthook
from .guicommon import align, Email, ClickableText, write_event_value_closure
from .version import version

def title(s):
    return sg.Text(s, font=("Helvetica", 16), justification='center')
def center_text(s, **kwargs):
    return sg.Text(s, justification='center', **kwargs)

def small_credits():
    def text(s, **kwargs):
        return sg.Text(s, font=("Helvetica", 8), pad=(0, 0), **kwargs)

    def bold(s, **kwargs):
        return sg.Text(s, font=("Helvetica Bold", 8), pad=(0, 0), **kwargs)

    return [
        sg.Column([
            [bold("Francisco Daniel Sturniolo")],
            [text("Desarrollador")],
            [Email("fd.sturniolo@gmail.com")]
        ]),
        sg.VerticalSeparator(),
        sg.Column([
            [bold("Dr. Leandro Bugnon")],
            [text("Director")],
            [Email("lbugnon@sinc.unl.edu.ar")]
        ]),
        sg.VerticalSeparator(),
        sg.Column([
            [bold("Dr. Julián Alberto Sabattini")],
            [text("Co-Director")],
            [Email("julian.sabattini@fca.uner.edu.ar")]
        ]),
    ]

def find_antlabeler() -> Path:
    labeler = Path("AntLabeler.exe")
    if not labeler.exists():
        raise FileNotFoundError
    return labeler

def main():
    from .loading_window import LoadingWindow

    sg.theme(C.THEME)
    with LoadingWindow():
        sys.excepthook = make_excepthook(Path.cwd())
        import matplotlib  # matplotlib is imported by pims by default

        matplotlib.use('agg')  # we set agg to avoid it using tk and risk multithreading issues
        print("loaded: ", matplotlib)

        import pims

        print("loaded: ", pims)
        from ..tracker import tracking

        print("loaded: ", tracking)
        from ..tracker import leafdetect

        det = leafdetect.TFLiteLeafDetector(C.TFLITE_MODEL, [])
        print("loaded: ", leafdetect)

    layout = [
        [align([
            [sg.Image(C.LOGO_AT)],
            [title(f"AntTracker v{version}")],

            [sg.HorizontalSeparator()],

            [center_text("Realizado en el marco del Proyecto Final de Carrera: \n"
                         "Desarrollo de una herramienta para identificación automática del ritmo de forrajeo\n"
                         " de hormigas cortadoras de hojas a partir de registros de video.")],
            [sg.HorizontalSeparator()],

            small_credits(),

            [sg.HorizontalSeparator()],

            [sg.Image(C.LOGO_FICH)],
            [sg.Image(C.LOGO_SINC),
             sg.Column([[sg.Image(C.LOGO_UNER)], [sg.Image(C.LOGO_AGRO)]],
                       element_justification='center')],
        ], 'center')],

        [align([[
            sg.Button("Avanzado", k='-ADVANCED-'),
            sg.Button("Más información", k='-MORE_INFO-'),
            sg.Button("Abrir carpeta de videos", k='-OPEN_FOLDER-', focus=True)]], 'right')]
    ]
    win = sg.Window("AntTracker", layout, icon=C.LOGO_AT_ICO, finalize=True)
    ClickableText.bind_all()
    while True:
        event, values = win.read()
        if event == sg.WIN_CLOSED:
            break
        if event == '-OPEN_FOLDER-':
            from .ant_tracker_routine import ant_tracker_routine

            win.disable()
            ant_tracker_routine()
            win.enable()
        if event == '-MORE_INFO-':
            about()
        if event == '-ADVANCED-':
            buttons = {
                '-ANTLABELER-': "AntLabeler",
                '-VALIDATOR-':  "Validador trk/tag",
                '-TRKVIZ-':     "Visualizador de trk/tag\n(experimental)",
            }
            adv_layout = [[align([
                *[[sg.Button(text, size=(20, 2), k=k)] for k, text in buttons.items()],
                [sg.HorizontalSeparator()],
                [sg.Button("Regresar", k='-BACK-')],
            ], 'center')]]
            adv_win = sg.Window("Avanzado", adv_layout, icon=C.LOGO_AT_ICO, modal=True)

            def wait_n_send(k):
                send = write_event_value_closure(adv_win)

                def _w():
                    from time import sleep
                    sleep(5)
                    send('!!' + k, '-OPEN_DONE-')

                import threading
                threading.Thread(target=_w, daemon=True).start()

            while True:
                event, values = adv_win.read()
                if event == sg.WIN_CLOSED or event == '-BACK-':
                    adv_win.close()
                    break
                if event.startswith('!!'):
                    key = event.split('!!')[1]
                    adv_win[key].update(buttons[key], disabled=False)
                if event == '-ANTLABELER-':
                    try:
                        p = find_antlabeler()
                    except FileNotFoundError:
                        sg.popup(C.ANTLABELER_UNAVAILABLE)
                        continue
                    adv_win[event].update("Abriendo...", disabled=True)
                    import os

                    os.startfile(p)
                    wait_n_send(event)
                if event == '-TRKVIZ-':
                    adv_win[event].update("Abriendo...", disabled=True)
                    from .trkviz import trkviz_subprocess

                    trkviz_subprocess()
                    wait_n_send(event)
                if event == '-VALIDATOR-':
                    from .validator import validate_routine

                    validate_routine()

    win.close()
