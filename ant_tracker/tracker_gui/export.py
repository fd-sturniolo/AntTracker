import datetime
import numpy as np
from functools import partial
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.worksheet.worksheet import Worksheet
from pathlib import Path
from typing import List, Iterable

from . import constants as C
from .session import SessionInfo
from ..tracker.info import TracksCompleteInfo, Direction
from ..tracker.track import Track

def EN(track: Track, info: TracksCompleteInfo):
    return info.track_direction(track) == Direction.EN

def SN(track: Track, info: TracksCompleteInfo):
    return info.track_direction(track) == Direction.SN

def loaded(tracks: Iterable[Track]):
    yield from (track for track in tracks if track.load_detected and track.load_prediction)

def unloaded(tracks: Iterable[Track]):
    yield from (track for track in tracks if track.load_detected and not track.load_prediction)

def onedim_conversion(length_or_speed: float, mm_per_pixel: float):
    """
    :param length_or_speed: in [px] or [px/s]
    :param mm_per_pixel: in [mm/px]
    :return: length i [mm] or speed in [mm/s]
    """
    return length_or_speed * mm_per_pixel

def area_conversion(area: float, mm_per_pixel: float):
    """
    :param area: in [px^2]
    :param mm_per_pixel: in [mm/px]
    :return: area in [mm^2]
    """
    return area * (mm_per_pixel ** 2)

def length(cell):
    if not cell.value: return 0
    if isinstance(cell.value, float):
        lc = len(str(round(cell.value, 2))) + 1
    elif isinstance(cell.value, datetime.datetime):
        lc = 19
    else:
        lc = len(str(cell.value)) + 1
    return lc

def adjust_column_widths(ws: Worksheet):
    dims = {}
    for row in ws.rows:
        for cell in row:
            if cell.row == 1: continue
            if cell.value:
                dims[cell.column_letter] = max(dims.get(cell.column_letter, 13), length(cell))
    for col, value in dims.items():
        ws.column_dimensions[col].width = value

    from openpyxl.utils import get_column_letter
    for rng in ws.merged_cells.ranges:
        cols = {}
        for cell in rng.cells:
            col = get_column_letter(cell[1])
            cols[col] = max(ws.column_dimensions[col].width, length(ws.cell(cell[1], cell[0])))
        total_length = sum(cols.values())
        for col in cols:
            ws.column_dimensions[col].width = total_length / len(cols)

def center_headers(ws: Worksheet, rows=1):
    for row in ws.iter_rows(1, rows):
        for cell in row:
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.row_dimensions[1].height = 30

def adjust_decimal_places(ws: Worksheet, decimal_places=2):
    for row in ws.rows:
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = '0' + ('.' + '0' * decimal_places) if decimal_places else ''

class Exporter:
    def __init__(self):
        self.__wb = None

    @property
    def workbook(self) -> Workbook:
        if self.__wb is None:
            raise AttributeError("Primero debe llamar a export(_progress)")
        return self.__wb

    def save(self, path: Path):
        self.workbook.save(path)

    def export(self, infos: List[TracksCompleteInfo], time_delta=datetime.timedelta(minutes=1)):
        list(self.export_progress(infos, time_delta))
        return self.__wb

    def export_progress(self, infos: List[TracksCompleteInfo], time_delta=datetime.timedelta(minutes=1)):
        yield "Inicializando...", 0, 1
        wb = Workbook()
        ws = wb.active

        # region Por hormiga
        ws.title = "Por hormiga"
        ws.append([
            "Video",
            "ID",
            "Dirección",
            "Carga",
            "Certeza en carga",
            "Velocidad prom. [mm/s]",
            "Área mediana [mm²]",
            "Largo mediana [mm]",
            "Ancho mediana [mm]",
            "Tiempo entrada",
            "Tiempo salida",
            "Frame inicio",
            "Frame final",
        ])

        progress_i = 0
        progress_max = sum([len(info.filter_tracks(**C.TRACKFILTER)) for info in infos])
        yield "Inicializando...", 1, 1
        yield "Generando análisis por hormiga", progress_i, progress_max
        for info in infos:
            od = partial(onedim_conversion, mm_per_pixel=info.mm_per_pixel)
            area = partial(area_conversion, mm_per_pixel=info.mm_per_pixel)
            for track in info.filter_tracks(**C.TRACKFILTER):
                ws.append([
                    info.video_name,
                    track.id,
                    "EN" if EN(track, info) else "SN" if SN(track, info) else "??",
                    ("Si" if track.load_prediction else "No") if track.load_detected else "??",
                    track.load_certainty if track.load_detected else 0,
                    od(track.speed_mean),
                    area(track.area_median),
                    od(track.length_median),
                    od(track.width_median),
                    info.time_at(track.first_frame()),
                    info.time_at(track.last_frame()),
                    track.first_frame(),
                    track.last_frame(),
                ])
                progress_i += 1
                yield "Generando análisis por hormiga", progress_i, progress_max

        # endregion

        # region Por video
        ws = wb.create_sheet("Por video")
        progress_i = 0
        progress_max = len(infos) * 16
        yield "Generando análisis por video", progress_i, progress_max

        ws.append(["Video",
                   "Hora Inicio",
                   "Hora Fin",
                   "Total hormigas entrando al nido (EN)", "",
                   "Total hormigas saliendo del nido (SN)", "",
                   "Velocidad promedio EN [mm/s]", "",
                   "Velocidad promedio SN [mm/s]", "",
                   "Área mediana EN [mm²]", "",
                   "Área mediana SN [mm²]", "",
                   "Largo mediana EN [mm]", "",
                   "Largo mediana SN [mm]", "",
                   "Ancho mediana EN [mm]", "",
                   "Ancho mediana SN [mm]", "",
                   ])
        ws.append(["", "", ""] + ["Cargadas", "Sin carga"] * 10)
        merge = ['A1:A2', 'B1:B2', 'C1:C2', 'D1:E1', 'F1:G1', 'H1:I1', 'J1:K1', 'L1:M1', 'N1:O1', 'P1:Q1', 'R1:S1',
                 'T1:U1', 'V1:W1']
        for m in merge: ws.merge_cells(m)
        props = ('speed_mean', 'area_median', 'length_median', 'width_median')

        for info in infos:
            filtered = info.filter_tracks(**C.TRACKFILTER)
            EN_tracks = [track for track in filtered if EN(track, info)]
            SN_tracks = [track for track in filtered if SN(track, info)]
            del filtered  # to save on memory

            od = partial(onedim_conversion, mm_per_pixel=info.mm_per_pixel)
            area = partial(area_conversion, mm_per_pixel=info.mm_per_pixel)

            data = {'EN': {'l': {}, 'u': {}}, 'SN': {'l': {}, 'u': {}}}
            for k1, tracks in zip(('EN', 'SN'), (EN_tracks, SN_tracks)):
                for k2, load in zip(('l', 'u'), (loaded, unloaded)):
                    for prop, conv in zip(props, (od, area, od, od)):
                        data[k1][k2][prop] = conv(np.mean([
                            getattr(track, prop) for track in load(tracks)
                        ]))
                        progress_i += 1
                        yield "Generando análisis por video", progress_i, progress_max

            ws.append([
                info.video_name,
                info.start_time,
                info.end_time,
                len(list(loaded(EN_tracks))),
                len(list(unloaded(EN_tracks))),
                len(list(loaded(SN_tracks))),
                len(list(unloaded(SN_tracks))),
                data['EN']['l']['speed_mean'],
                data['EN']['u']['speed_mean'],
                data['SN']['l']['speed_mean'],
                data['SN']['u']['speed_mean'],
                data['EN']['l']['area_median'],
                data['EN']['u']['area_median'],
                data['SN']['l']['area_median'],
                data['SN']['u']['area_median'],
                data['EN']['l']['length_median'],
                data['EN']['u']['length_median'],
                data['SN']['l']['length_median'],
                data['SN']['u']['length_median'],
                data['EN']['l']['width_median'],
                data['EN']['u']['width_median'],
                data['SN']['l']['width_median'],
                data['SN']['u']['width_median'],
            ])

        lastrow = len(infos) + 2
        ws.append(["Total", f"=MIN(B3:B{lastrow})", f"=MAX(C3:C{lastrow})"] +
                  [f"=SUM({c}3:{c}{lastrow})" for c in "DEFG"] +
                  [f"=AVERAGE({c}3:{c}{lastrow})" for c in "HIJKLMNOPQRSTUVW"])
        ws.cell(lastrow + 1, 2).number_format = 'yyyy-mm-dd h:mm:ss'
        ws.cell(lastrow + 1, 3).number_format = 'yyyy-mm-dd h:mm:ss'

        # endregion

        # region En el tiempo
        ws = wb.create_sheet("En el tiempo")
        ws.append([
            "Hora Inicio",
            "Hora Fin",
            "Total hormigas entrando al nido (EN)", "",
            "Total hormigas saliendo del nido (SN)", "",
            "Velocidad promedio EN [mm/s]", "",
            "Velocidad promedio SN [mm/s]", "",
            "Área mediana EN [mm²]", "",
            "Área mediana SN [mm²]", "",
            "Largo mediana EN [mm]", "",
            "Largo mediana SN [mm]", "",
            "Ancho mediana EN [mm]", "",
            "Ancho mediana SN [mm]", "",
        ])
        ws.append(["", ""] + ["Cargadas", "Sin carga"] * 10)
        merge = ['A1:A2', 'B1:B2', 'C1:D1', 'E1:F1', 'G1:H1', 'I1:J1', 'K1:L1', 'M1:N1', 'O1:P1', 'Q1:R1', 'S1:T1',
                 'U1:V1']
        for m in merge: ws.merge_cells(m)

        start_time = min([info.start_time for info in infos])
        end_time = max([info.end_time for info in infos])

        progress_i = 0
        progress_max = (end_time - start_time) // time_delta + 1
        yield "Generando análisis en el tiempo", progress_i, progress_max
        time = start_time
        while time < end_time:
            totals = {'en-load': 0, 'en-ntld': 0, 'sn-load': 0, 'sn-ntld': 0}
            speeds = {'en-load': [], 'en-ntld': [], 'sn-load': [], 'sn-ntld': []}
            areas = {'en-load': [], 'en-ntld': [], 'sn-load': [], 'sn-ntld': []}
            lengths = {'en-load': [], 'en-ntld': [], 'sn-load': [], 'sn-ntld': []}
            widths = {'en-load': [], 'en-ntld': [], 'sn-load': [], 'sn-ntld': []}
            for info in infos:
                _filter = info.filter_func(**C.TRACKFILTER)
                tracks = [track for track in info.tracks_in_time(time, time + time_delta)
                          if track.load_detected and _filter(track)]
                if tracks:
                    od = partial(onedim_conversion, mm_per_pixel=info.mm_per_pixel)
                    area = partial(area_conversion, mm_per_pixel=info.mm_per_pixel)

                    for track in tracks:
                        if track.load_prediction and EN(track, info):
                            key = 'en-load'
                        elif not track.load_prediction and EN(track, info):
                            key = 'en-ntld'
                        elif track.load_prediction and SN(track, info):
                            key = 'sn-load'
                        elif not track.load_prediction and SN(track, info):
                            key = 'sn-ntld'
                        else:
                            continue

                        totals[key] += 1
                        speeds[key].append(od(track.speed_mean))
                        areas[key].append(area(track.area_median))
                        lengths[key].append(od(track.length_median))
                        widths[key].append(od(track.width_median))

            ws.append([
                time,
                time + time_delta,
                totals['en-load'],
                totals['en-ntld'],
                totals['sn-load'],
                totals['sn-ntld'],
                np.mean(speeds['en-load']),
                np.mean(speeds['en-ntld']),
                np.mean(speeds['sn-load']),
                np.mean(speeds['sn-ntld']),
                np.mean(areas['en-load']),
                np.mean(areas['en-ntld']),
                np.mean(areas['sn-load']),
                np.mean(areas['sn-ntld']),
                np.mean(lengths['en-load']),
                np.mean(lengths['en-ntld']),
                np.mean(lengths['sn-load']),
                np.mean(lengths['sn-ntld']),
                np.mean(widths['en-load']),
                np.mean(widths['en-ntld']),
                np.mean(widths['sn-load']),
                np.mean(widths['sn-ntld']),
            ])
            time += time_delta
            progress_i += 1
            yield "Generando análisis en el tiempo", progress_i, progress_max
        # endregion

        for i_ws, ws in enumerate(wb.worksheets):
            adjust_column_widths(ws)
            adjust_decimal_places(ws)
            if i_ws in (0, 2):
                center_headers(ws, 2)
            else:
                center_headers(ws)

        # Los datos se calculan en órden: hormiga-video-tiempo, porque 'hormiga' indica
        # progreso más rápido que las demás y da a entender más rápidamente que hay trabajo
        # en proceso. Sin embargo, el órden de hojas tiene que ser: video-hormiga-tiempo
        wb.move_sheet("Por video", -1)

        self.__wb = wb
        yield "Finalizado", 1, 1

if __name__ == '__main__':
    sesspath = Path("../tracker/vid_tags/Prueba 1 AntTracker 4-Dic-2020/.anttrackersession")
    session = SessionInfo.load(sesspath)
    trkfiles = [session.get_trkfile(f) for f in session.videofiles]
    e = Exporter()
    for t, i, mx in e.export_progress([TracksCompleteInfo.load(f) for f in trkfiles]):
        print(t, f"{i}/{mx}")

    file = sesspath.parent / "export.xlsx"
    while True:
        try:
            e.save(file)
            break
        except PermissionError:
            input(f"El archivo {file} está abierto o protegido. Presione Enter para probar nuevamente")
