import traceback

import PySimpleGUI as sg
import datetime
import re
import sys
from pathlib import Path

from . import constants as C
from .guicommon import align
from .loading_window import LoadingWindow

def make_excepthook(path: Path, in_routine=False):
    def excepthook(exc_type, exc_value, exc_tb):
        LoadingWindow.close_all()
        lines = traceback.format_exception(exc_type, exc_value, exc_tb)

        def shorten(match):
            _path = Path(match.group(1))
            relevant_parts = []
            for p in _path.parts[::-1]:
                relevant_parts.append(p)
                if p in ("envs", "Miniconda3", "Lib", "site-packages", "ant_tracker"):
                    break
            return f'File "{Path(*relevant_parts[::-1])}"'

        # if PyInstaller-frozen, shorten paths for privacy
        if C.FROZEN:
            lines = [re.sub(r'File "([^"]+)"', shorten, line) for line in lines]

        tb = "".join(lines)
        print(tb)
        filename = str(path / ("error-" + datetime.datetime.now(tz=None).strftime('%Y-%m-%dT%H_%M_%S') + ".log"))
        with open(filename, "w") as f:
            f.write(tb)
        import textwrap
        mlwidth = 100
        mlheight = 15
        w = sg.Window("Error", [
            [sg.Text(f"Se produjo un error. El siguiente archivo contiene los detalles:")],
            [sg.InputText(filename, disabled=True, k='-FILE-')],
            [sg.Text(textwrap.fill(
                "Por favor envíe el mismo " + (
                    "y, en lo posible, los archivos "
                    ".anttrackersession y .trk en la misma carpeta " if in_routine else "") +
                "a la persona que le proporcionó este programa.", 100))],
            [sg.Multiline(tb, size=(mlwidth, mlheight), disabled=True, k='-D-', visible=False)],
            [align(sg.Button("Ver detalle", k='-DB-'), 'left'), align(sg.CloseButton("Cerrar"), 'right')],
        ], finalize=True)
        w['-FILE-'].expand(expand_x=True)
        detail_visible = False
        while True:
            event, values = w.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == '-DB-':
                x, y = w.CurrentLocation()
                w.move(x - int(mlwidth * 1.5), y - mlheight * 10)
                detail_visible = not detail_visible
                w['-D-'].update(visible=detail_visible)
                w['-DB-'].update(visible=False)
        sys.exit(0)

    return excepthook
