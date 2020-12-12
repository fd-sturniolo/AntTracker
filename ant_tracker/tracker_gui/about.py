import PySimpleGUI as sg

from . import constants as C
from .guicommon import transparent_multiline, Email, Link, ClickableText

def about(width=90, height=20):
    def text(s, **kwargs):
        return sg.Text(s, font=("Helvetica", 8), pad=(0, 0), **kwargs)

    def bold(s, **kwargs):
        return sg.Text(s, font=("Helvetica Bold", 8), justification='center', pad=(0, 0), **kwargs)

    creds = [
        sg.Column([
            [bold("Francisco Daniel Sturniolo")],
            [text("Desarrollador")],
            [text("Facultad de Ingeniería y Ciencias Hídricas")],
            [text("Universidad Nacional del Litoral")],
            [text("Santa Fe, Santa Fe, Argentina")],
            [Email("fd.sturniolo@gmail.com")],
        ]),
        sg.VerticalSeparator(),
        sg.Column([
            [bold("Dr. Leandro Bugnon")],
            [text("Director")],
            [text("Research Institute for Signals, Systems and\nComputational Intelligence, sinc(i)")],
            [text("(FICH-UNL/CONICET)")],
            [text("Ciudad Universitaria")],
            [text("Santa Fe, Santa Fe, Argentina")],
            [Email("lbugnon@sinc.unl.edu.ar")],
            [Link("www.sinc.unl.edu.ar", linktext="sinc(i)")],
        ]),
        sg.VerticalSeparator(),
        sg.Column([
            [bold("Dr. Julián Alberto Sabattini")],
            [text("Co-Director")],
            [text("Ecology Agricultural Systems")],
            [text("Faculty of Agricultural Sciences")],
            [text("National University of Entre Rios")],
            [text("Route No. 11 km 10.5")],
            [text("Oro Verde, Entre Ríos (Argentina)")],
            [Email("julian.sabattini@fca.uner.edu.ar")],
            [Link("https://www.researchgate.net/profile/Julian_Sabattini", linktext="ResearchGate")],
            [text("Skype: juliansabattini")],
        ]),
    ]
    layout = [
        creds,
        [sg.HorizontalSeparator()],
        [transparent_multiline(C.ABOUT_INFO, width, height)],
        [sg.Text("AntTracker es código abierto (licencia MIT) y puede encontrarse en:"),
         Link("http://github.com/fd-sturniolo/anttracker", font=("Helvetica", 10))],
    ]
    win = sg.Window("Sobre AntTracker", layout, icon=C.LOGO_AT_ICO,
                    disable_minimize=True, modal=True, finalize=True)
    ClickableText.bind_all()
    while True:
        e, _ = win.read()
        if e == sg.WIN_CLOSED:
            win.close()
            break
