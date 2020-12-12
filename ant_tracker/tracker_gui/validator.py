import PySimpleGUI as sg
from pathlib import Path

from . import constants as C
from .guicommon import align
from .loading_window import LoadingWindow
from ..tracker.validate import Measures

XLSX = '.xlsx'

def validate_routine():
    def file_open(description, file_types, k, default="", save=False, **browse_kwargs):
        browse_cls = sg.FileBrowse if not save else sg.FileSaveAs
        return sg.Column([
            [sg.Text(description, pad=(0, 0))],
            [sg.Input(default_text=default, pad=(3, 0), k=k),
             browse_cls("Examinar", file_types=file_types, pad=(3, 0), **browse_kwargs)]
        ])

    layout = [
        [file_open("Archivo de tracking:", file_types=(("Tracking", "*.trk"),), k='-TRK-')],
        [file_open("Archivo de etiquetas:", file_types=(("Etiquetas", "*.tag"),), k='-TAG-')],
        [align(sg.Ok("Validar"), 'right')],
    ]
    window = sg.Window("Seleccionar archivos", layout, modal=True, icon=C.LOGO_AT_ICO, finalize=True)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == 'Validar':
            try:
                trk, tag = Path(values['-TRK-']), Path(values['-TAG-'])
            except:
                sg.popup_error("Hubo un error al leer los archivos. "
                               "Asegúrese de haber proporcionado las rutas correctas.")
                continue
            if trk.suffix != '.trk' or tag.suffix != '.tag':
                sg.popup_error("Debe cargar todos los archivos y asegurarse de que sus extensiones sean correctas.")
                continue
            else:
                from functools import partial
                Load = partial(LoadingWindow, spinner=C.SMALLSPINNER)
                with Load("Cargando tracking..."):
                    from ..tracker.info import TracksCompleteInfo, reposition_into_crop
                    tracked = TracksCompleteInfo.load(trk)
                with Load("Cargando etiquetas..."):
                    from ..tracker.ant_labeler_info import LabelingInfo
                    truth = reposition_into_crop(LabelingInfo.load(tag), tracked.crop_rect)
                if tracked.video_hash != truth.video_hash:
                    sg.popup_error("Los archivos corresponden a videos distintos.")
                    continue
                with Load("Validando..."):
                    from ..tracker.validate import export_like_measures
                    measures = export_like_measures(truth, tracked, trackfilter=C.TRACKFILTER)
                    wb = make_xl(measures)

                while True:
                    try:
                        exportpath = (tag.parent / 'valid').with_suffix(XLSX)

                        def get_exportpath(prev_path):
                            return Path(sg.Window(
                                "Validación - Archivo de salida", [
                                    [file_open("Archivo de salida", (("Excel", '*.xlsx'),), '-', str(prev_path), True,
                                               initial_folder=tag.parent, default_extension=XLSX)],
                                    [align(sg.Ok(), 'right')]
                                ], icon=C.LOGO_AT_ICO, disable_close=True, modal=True).read(close=True)[1]['-'])

                        exportpath = get_exportpath(exportpath).with_suffix(XLSX)
                        while not exportpath.parent.exists() and not exportpath.is_absolute():
                            sg.popup_error("Debe asegurarse de que la ruta de salida sea válida.")
                            exportpath = get_exportpath(exportpath).with_suffix(XLSX)

                        wb.save(exportpath)
                        break
                    except PermissionError:
                        sg.popup_error("El archivo está abierto o protegido de alguna manera.\n"
                                       "Intente cerrar el programa que mantiene el archivo abierto o "
                                       "guardarlo con otro nombre.")

                window.close()
                break

def make_xl(measures: Measures):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    import dataclasses
    labeldict = dataclasses.asdict(measures.label)
    trackdict = dataclasses.asdict(measures.track)
    descriptions = {
        'total_EN':      "Total EN",
        'total_SN':      "Total SN",
        'speed_mean':    "Velocidad prom. [px/s]",
        'area_median':   "Área mediana [px²]",
        'length_median': "Largo mediana [px]",
        'width_median':  "Ancho mediana [px]",
    }

    ws.append(["Medidas", "Etiquetas", "Tracks", "Error relativo"])
    for k in labeldict.keys():
        error = abs(labeldict[k] - trackdict[k]) / labeldict[k] if labeldict[k] != 0 else 'N/A'
        ws.append([descriptions[k], labeldict[k], trackdict[k], error])
    from .export import adjust_column_widths, adjust_decimal_places
    adjust_column_widths(ws)
    adjust_decimal_places(ws, 3)
    for rows in ws['D2:D7']:
        for cell in rows:
            cell.number_format = '0.00%'
    ws.append([])
    ws.append(["Aclaración: cada una de las medidas es un promedio a lo largo de todas las hormigas."])
    ws.append(["eg: Velocidad prom. es el promedio de las velocidades promedio de todas las hormigas."])
    ws.append(["Además para valores de la columna 'Etiquetas' iguales a 0, no se calcula el error."])
    ws.append(["En este caso, asegúrese de etiquetar más hormigas para obtener muestras representativas."])

    return wb

if __name__ == '__main__':
    validate_routine()
