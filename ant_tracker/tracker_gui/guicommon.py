import PySimpleGUI as sg
from typing import Union, Literal, List, Dict

def write_event_value_closure(window):
    def send(event, value):
        try:
            window.write_event_value(event, value)
        except RuntimeError as e:
            # este error ocurre intermitentemente, y la única manera de reproducirlo más o menos consistentemente es
            # abrir un popup en el main thread mientras se hacen llamadas a write_event_value.
            # hay un juego de threads que PySimpleGUI no supo resolver y que causa el error, pero generalmente no
            # suele haber problema si ignoramos el error y hacemos la llamada de vuelta
            print("Ignorado: ", repr(e))
            print("Intentando nuevamente...")
            send(event, value)

    return send

class ClickableText(sg.Text):
    _to_bind: Dict[str, 'ClickableText'] = {}

    def __init__(self, s, k, font=("Helvetica", 8), **kwargs):
        super(ClickableText, self).__init__(s, justification='center', pad=(0, 0), font=font, text_color="blue", k=k,
                                            **kwargs)
        ClickableText._to_bind[self.Key] = self
        self.action = lambda e: print(f"No action set for: {self}")
        self.bound = False

    @classmethod
    def bind_all(cls):
        for key, elem in cls._to_bind.items():
            if not elem.bound:
                elem.bind_self()
                elem.bound = True

    def bind_self(self):
        def _underline(elem, set_to: bool):
            original_font = elem.TKText['font']

            def underline(_):
                if set_to:
                    elem.TKText.configure(font=original_font + ' underline')
                else:
                    elem.TKText.configure(font=original_font)

            return underline

        self.Widget.bind('<Button-1>', self.action)
        self.Widget.bind('<Enter>', _underline(self, True))
        self.Widget.bind('<Leave>', _underline(self, False))
        self.set_cursor("hand2")

class Link(ClickableText):
    """Clickable link. Must call ``ClickableText.bind_all`` after finalizing the containing Window"""

    def __init__(self, link, font=("Helvetica", 8), linktext=None, **kwargs):
        if linktext is None:
            linktext = link
        key = None
        if 'k' in kwargs:
            key = kwargs['k']
            del kwargs['k']
        if 'key' in kwargs:
            key = kwargs['key']
            del kwargs['key']
        if key is None: key = f"__ANTTRACKER__!LINK_{linktext}"
        super(Link, self).__init__(linktext, font=font, k=key, **kwargs)

        def _goto(_link):
            def goto(_):
                import webbrowser
                webbrowser.open(_link)

            return goto

        self.action = _goto(link)

class Email(Link):
    """Clickable e-mail link. Must call ``ClickableText.bind_all`` after finalizing the containing Window"""

    def __init__(self, email, font=("Helvetica", 8), **kwargs):
        super(Email, self).__init__(f"mailto:{email}", linktext=email, font=font, k=f"__ANTTRACKER__!EMAIL_{email}",
                                    **kwargs)

def transparent_multiline(text, width, height, **kwargs):
    return sg.Multiline(text, size=(width, height), disabled=True, write_only=True, background_color="#f2f2f2",
                        border_width=0, auto_size_text=False, **kwargs)

def align(layout_or_elem: Union[List[List[sg.Element]], sg.Element], justification: Literal['left', 'right', 'center']):
    if not isinstance(layout_or_elem, list):
        layout_or_elem = [[layout_or_elem]]
    return sg.Column(layout_or_elem, element_justification=justification, expand_x=True)

def parse_number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def release(key): return key + "+UP"

def change_bar_color(progbar: sg.ProgressBar, color: str, background: str = None):
    from tkinter import ttk
    s = ttk.Style()
    if background:
        s.configure(progbar.TKProgressBar.style_name, background=color, troughcolor=background)
    else:
        s.configure(progbar.TKProgressBar.style_name, background=color)
