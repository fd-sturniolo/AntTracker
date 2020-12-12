"""Constants"""
import sys

FROZEN = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

TFLITE_MODEL = "model.tflite"
SPINNER = "images/spinner.gif"
SMALLSPINNER = "images/small spinner.gif"
LOGO_FICH = "images/logo_fich.png"
LOGO_SINC = "images/logo_sinc.png"
LOGO_UNER = "images/logo_uner.png"
LOGO_AGRO = "images/logo_agro.png"
LOGO_AT = "images/icon.png"
LOGO_AT_ICO = "images/icon.ico"
if not FROZEN:
    from pathlib import Path

    this_dir = Path(__file__).parent
    TFLITE_MODEL = str(this_dir / TFLITE_MODEL)
    SPINNER = str(this_dir / SPINNER)
    SMALLSPINNER = str(this_dir / SMALLSPINNER)
    LOGO_FICH = str(this_dir / LOGO_FICH)
    LOGO_SINC = str(this_dir / LOGO_SINC)
    LOGO_UNER = str(this_dir / LOGO_UNER)
    LOGO_AGRO = str(this_dir / LOGO_AGRO)
    LOGO_AT = str(this_dir / LOGO_AT)
    LOGO_AT_ICO = str(this_dir / LOGO_AT_ICO)

THEME = 'Default1'

RESP_SI = "  Sí  "
RESP_NO = "  No  "

ANTLABELER_UNAVAILABLE = "AntLabeler no pudo ser encontrado."

TRACKFILTER = {'filter_center_center': True, 'length_of_tracks': 5}

def format_triple_quote(s):
    if s[0] == "\n": s = s[1:]
    return s.replace("\n\n\n", "-SPACE-") \
        .replace("\n\n", "-NL-") \
        .replace("\n", " ") \
        .replace("  ", " ") \
        .replace("-NL-", "\n") \
        .replace("-SPACE-", "\n\n") \
        .replace("\n ", "\n")

# Usar dos saltos de línea para separar en párrafos. Para dejar un espacio entre párrafos, usar tres saltos de línea.
ABOUT_INFO = format_triple_quote("""
Desarrollado durante 2019-2020 por Francisco Daniel Sturniolo,
en el marco de su Proyecto Final de Carrera para el título de Ingeniero en Informática
de la Facultad de Ingeniería y Ciencias Hídricas de la Universidad Nacional del Litoral,
bajo la dirección de Leandro Bugnon y la co-dirección de Julián Sabattini, 
titulado "Desarrollo de una herramienta para identificación automática del ritmo de forrajeo
de hormigas cortadoras de hojas a partir de registros de video".


El mismo pretende analizar el comportamiento de forrajeo de las HCH a partir de videos tomados de la salida de un
hormiguero (tales como los obtenidos a partir del dispositivo AntVRecord), detectando las trayectorias tomadas por las
hormigas y su posible carga de hojas, para luego extraer estadísticas temporales de su comportamiento
y volumen vegetal recolectado.


También incluido con este programa se encuentra AntLabeler, una utilidad de etiquetado para videos de la misma índole, 
que fue utilizada para validar los resultados obtenidos por AntTracker sobre videos de prueba. El uso de esta
herramienta actualmente se encuentra supercedido por AntTracker, pero se provee como una forma de revisar con precisión
las trayectorias y cargas detectadas.



This software uses libraries from the FFmpeg project under the LGPLv2.1.
""")
