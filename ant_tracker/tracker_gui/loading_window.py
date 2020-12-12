from typing import List

from . import constants as C

class FakeLoadTask:
    def __init__(self, message, spinner):
        self.message = message
        self.spinner = spinner
        self._running = True

    def finish(self):
        self._running = False

    def run(self):
        import PySimpleGUI as sg
        sg.theme('Default1')
        while self._running:
            sg.popup_animated(self.spinner, self.message, time_between_frames=100)
        sg.popup_animated(None)
        return

def fake_loader(message, spinner):
    from multiprocessing import Process
    task = FakeLoadTask(message, spinner)
    p = Process(target=task.run, daemon=True)

    class FakeLoader:
        @staticmethod
        def finish():
            nonlocal task, p
            task.finish()
            p.kill()

    p.start()
    return FakeLoader()

class LoadingWindow:
    __windows: List['LoadingWindow'] = []

    def __init__(self, message="Cargando...", spinner=C.SPINNER):
        self.message = message
        self.spinner = spinner
        LoadingWindow.__windows.append(self)

    def __enter__(self):
        self.fl = fake_loader(self.message, self.spinner)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fl.finish()

    @classmethod
    def close_all(cls):
        for window in cls.__windows:
            window.fl.finish()
            del window
        cls.__windows = []
