from dataclasses import dataclass, asdict
from typing import ClassVar, Dict

@dataclass
class SegmenterParameters:
    gaussian_sigma: float
    minimum_ant_radius: int
    movement_detection_history: int
    discard_percentage: float
    movement_detection_threshold: int
    approx_tolerance: float

    doh_min_sigma: float
    doh_max_sigma: float
    doh_num_sigma: int

    name_map: ClassVar = {
        'gaussian_sigma':               "Sigma de gaussiana",
        'minimum_ant_radius':           "Radio mínimo detectable",
        'movement_detection_history':   "Historia detección de fondo",
        'discard_percentage':           "Porcentaje para descarte",
        'movement_detection_threshold': "Umbral de detec. de movimiento",
        'approx_tolerance':             "Tolerancia en aproximación",
        'doh_min_sigma':                "Mínimo sigma DOH",
        'doh_max_sigma':                "Máximo sigma DOH",
        'doh_num_sigma':                "Número de valores de sigma",
    }
    description_map: ClassVar = {
        'gaussian_sigma':               "Nivel de borroneado de imagen para detección de objetos. A mayor valor, "
                                        "más grandes y circulares las detecciones, pero menos detecciones espurias.",
        'minimum_ant_radius':           "En píxeles, el radio mínimo que ocupa una hormiga. Hormigas "
                                        "más chicas no serán detectadas. Mismos efectos que el parámetro anterior.",
        'movement_detection_history':   "Número de cuadros almacenados para detección de fondo. A mayor cantidad, "
                                        "aumenta la capacidad de distinguir el fondo de las hormigas, pero aumenta "
                                        "el tiempo de procesamiento.",
        'discard_percentage':           "Sensibilidad a movimientos de cámara. Si se detecta que más de X% del cuadro "
                                        "es distinto al anterior, las detecciones se descartan.",
        'movement_detection_threshold': "Valor por sobre el cual se determina que un píxel está en movimiento. "
                                        "A mayor valor, más sensible al movimiento.",
        'approx_tolerance':             "Tolerancia al simplificar las formas detectadas. A menor valor, mayor "
                                        "precisión en la forma (aunque casi insignificante), "
                                        "pero mucho mayor tamaño de archivos generados "
                                        "y mayor tiempo de procesamiento.",
        'doh_min_sigma':                "Mínimo valor de sigma para el algoritmo DOH. Mientras menor sea, "
                                        "mayor probabilidad de encontrar hormigas pequeñas.",
        'doh_max_sigma':                "Máximo valor de sigma para el algoritmo DOH. Mientras mayor sea, "
                                        "mayor probabilidad de detectar objetos grandes como una sola hormiga.",
        'doh_num_sigma':                "Cantidad de valores intermedios entre mínimo y máximo que se consideran "
                                        "en DOH. Mientras más valores, mejor precisión, "
                                        "pero mayor tiempo de procesamiento.",
    }

    def __init__(self, params: Dict = None, **kwargs):
        if params is None:
            params = {}

        def kwarg_or_dict(key):
            d = params.get(key, None)
            if d is not None:
                return d
            d = kwargs.get(key, None)
            return d

        self.gaussian_sigma = kwarg_or_dict('gaussian_sigma')
        self.minimum_ant_radius = kwarg_or_dict('minimum_ant_radius')
        self.movement_detection_history = kwarg_or_dict('movement_detection_history')
        self.discard_percentage = kwarg_or_dict('discard_percentage')
        self.movement_detection_threshold = kwarg_or_dict('movement_detection_threshold')
        self.approx_tolerance = kwarg_or_dict('approx_tolerance')
        self.doh_min_sigma = kwarg_or_dict('doh_min_sigma')
        self.doh_max_sigma = kwarg_or_dict('doh_max_sigma')
        self.doh_num_sigma = kwarg_or_dict('doh_num_sigma')

    def values(self):
        return [v for k, v in self.items()]

    def keys(self):
        return [k for k, v in self.items()]

    def items(self):
        return [(k, v) for k, v in asdict(self).items() if v is not None]

    def names(self):
        return [v for k, v in self.name_map.items() if asdict(self)[k] is not None]

    def descriptions(self):
        return [v for k, v in self.description_map.items() if asdict(self)[k] is not None]

    def encode(self):
        return dict(self.items())

    @classmethod
    def decode(cls, serial):
        return cls(serial)

    @classmethod
    def mock(cls):
        return SegmenterParameters()

# noinspection PyPep8Naming
def LogWSegmenterParameters(params=None):
    if params is None:
        params = {}
    return SegmenterParameters({
        **{
            "gaussian_sigma":               8,
            "minimum_ant_radius":           10,
            "movement_detection_history":   50,
            "discard_percentage":           .3,
            "movement_detection_threshold": 25,
            "approx_tolerance":             1,
        }, **params
    })

# noinspection PyPep8Naming
def DohSegmenterParameters(params=None):
    if params is None:
        params = {}
    return SegmenterParameters({
        **{
            "gaussian_sigma":               8,
            "minimum_ant_radius":           10,
            "movement_detection_history":   50,
            "discard_percentage":           .3,
            "movement_detection_threshold": 25,
            "approx_tolerance":             1,
            "doh_min_sigma":                6,
            "doh_max_sigma":                30,
            "doh_num_sigma":                4,
        }, **params
    })

@dataclass
class TrackerParameters:
    max_distance_between_assignments: int
    frames_until_close: int
    a_sigma: float
    defaults: ClassVar = {
        "max_distance_between_assignments": 30,
        "frames_until_close":               8,
        "a_sigma":                          0.05,
    }
    name_map: ClassVar = {
        'max_distance_between_assignments': "Distancia entre asignaciones",
        'frames_until_close':               "N° de cuadros para cerrar trayecto",
        'a_sigma':                          "Peso del componente de aceleración",
    }
    description_map: ClassVar = {
        'max_distance_between_assignments': "Distancia máxima (en píxeles) a la que se puede asignar "
                                            "una detección en un cuadro a otra en el cuadro siguiente. "
                                            "A mayor número, mayores chances de recuperarse ante detecciones "
                                            "fallidas, pero aumentan las posibilidades de error en tracking.",
        'frames_until_close':               "Número de cuadros, desde que se pierde el seguimiento de un trayecto, "
                                            "hasta que efectivamente se da por perdido.",
        'a_sigma':                          "Peso del componente de aceleración, en cualquier dirección, "
                                            "en el modelo de tracking. A mayor valor, el algoritmo asume cada vez "
                                            "más errático el movimiento de las hormigas.",
    }

    def __init__(self, params: Dict = None, use_defaults=False, **kwargs):
        if params is None:
            params = {}

        def kwarg_or_dict_or_default(key):
            d = params.get(key, None)
            if d is not None: return d
            d = kwargs.get(key, None)
            if d is not None: return d
            if use_defaults: return self.defaults[key]
            return None

        self.max_distance_between_assignments = kwarg_or_dict_or_default('max_distance_between_assignments')
        self.frames_until_close = kwarg_or_dict_or_default('frames_until_close')
        self.a_sigma = kwarg_or_dict_or_default('a_sigma')

    def values(self):
        return [v for k, v in self.items()]

    def keys(self):
        return [k for k, v in self.items()]

    def items(self):
        return [(k, v) for k, v in asdict(self).items() if v is not None]

    def names(self):
        return [v for k, v in self.name_map.items() if asdict(self)[k] is not None]

    def descriptions(self):
        return [v for k, v in self.description_map.items() if asdict(self)[k] is not None]

    def name_desc_values(self):
        return list(zip(self.names(), self.descriptions(), self.values()))

    def encode(self):
        return dict(self.items())

    @classmethod
    def decode(cls, serial):
        return cls(serial)

    @classmethod
    def mock(cls):
        return TrackerParameters()
