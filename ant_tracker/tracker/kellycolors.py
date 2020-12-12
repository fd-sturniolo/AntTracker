from typing import List

from .common import Color

class KellyColors:
    index = 0

    def __init__(self):
        super().__init__()

    @classmethod
    def next(cls) -> Color:
        ret = cls.all()[cls.index]
        cls.index = (cls.index + 1) % len(cls.all())
        return ret

    @classmethod
    def get(cls, index: int) -> Color:
        """Gets a modulo-indexed color"""
        return cls.all()[index % len(cls.all())]

    @classmethod
    def all(cls) -> List[Color]:
        """Returns all kelly colors"""
        return [
            (255, 179, 0),
            (128, 62, 117),
            (255, 104, 0),
            (166, 189, 215),
            (193, 0, 32),
            (206, 162, 98),
            (129, 112, 102),
            (0, 125, 52),
            (246, 118, 142),
            (0, 83, 138),
            (255, 122, 92),
            (83, 55, 122),
            (255, 142, 0),
            (179, 40, 81),
            (244, 200, 0),
            (127, 24, 13),
            (147, 170, 0),
            (89, 51, 21),
            (241, 58, 19),
            (35, 44, 22)
        ]
