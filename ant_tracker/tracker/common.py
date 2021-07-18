from enum import Enum, auto

import numpy as np
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, NamedTuple, NewType, Sequence, Tuple, TypeVar, Union, cast, Any, Generator, Iterable

from .prettyjson import prettyjson

Color = Union[Tuple[int, int, int], Tuple[float, float, float]]
BinaryMask = np.ndarray
ColorImage = NewType('ColorImage', np.ndarray)
GrayscaleImage = NewType('GrayscaleImage', np.ndarray)
Image_T = Union[ColorImage, GrayscaleImage]
try:
    # noinspection PyUnresolvedReferences
    from pims import FramesSequence
    # noinspection PyUnresolvedReferences
    from slicerator import Pipeline

    Video = Union[FramesSequence, Sequence[ColorImage], Pipeline]
except ImportError:
    Video = Sequence[ColorImage]
NpPosition = np.ndarray
Vector = np.ndarray
Contour = np.ndarray
FrameNumber = int
Pixel = np.ndarray

class Position(NamedTuple):
    x: int
    y: int

    @property
    def yx(self):
        return self.y, self.x

    def distance_to(self, other):
        from math import sqrt
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"(x={self.x}, y={self.y})"

class Rect(NamedTuple):
    """A rectangle in a coordinate system with (0,0) at top-left and (xmax, ymax) at bottom-right."""
    x0: int
    x1: int
    y0: int
    y1: int

    @classmethod
    def from_points(cls, one: Tuple[int, int], other: Tuple[int, int]):
        """Creates an instance from two 2ples. The resulting `Rect` has the coordinates ordered so that
         `(x0,y0)` is top-left and `(x1,y1)` is bottom-right."""
        fx, fy = one
        lx, ly = other
        x0 = min(fx, lx)
        x1 = max(fx, lx)
        y0 = min(fy, ly)
        y1 = max(fy, ly)
        return cls(x0=x0, x1=x1, y0=y0, y1=y1)

    @property
    def xxyy(self):
        return self.x0, self.x1, self.y0, self.y1

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def diagonal_length(self):
        from math import sqrt
        return sqrt(self.height ** 2 + self.width ** 2)

    @property
    def center(self):
        return Position(x=self.x0 + (self.x1 - self.x0) // 2, y=self.y0 + (self.y1 - self.y0) // 2)

    @property
    def topleft(self):
        return Position(x=self.x0, y=self.y0)

    @property
    def bottomright(self):
        return Position(x=self.x1, y=self.y1)

    def __contains__(self, point: Union[Tuple[int, int], Position]):
        """`point` in `self`"""
        if isinstance(point, Position):
            x = point.x
            y = point.y
        else:
            x = point[0]
            y = point[1]
        return (self.topleft.x < x < self.bottomright.x) and (self.topleft.y < y < self.bottomright.y)

    def scale(self, imshape: Tuple[int, int], *, factor: float = None, extra_pixels: int = None):
        """Scale the rect by `factor` or add `extra_pixels` to each side"""
        if (factor is None) == (extra_pixels is None):
            raise ValueError("Only factor or extra_pixels, not both")
        elif factor is not None:
            w = self.width * factor
            h = self.height * factor
            return Rect(
                x0=int(self.center.x - w / 2),
                x1=int(self.center.x + w / 2 + 1),
                y0=int(self.center.y - h / 2),
                y1=int(self.center.y + h / 2 + 1),
            ).__bring_back_in(imshape).clip(imshape)
        elif extra_pixels is not None:
            return Rect(
                x0=int(self.x0 - extra_pixels),
                x1=int(self.x1 + extra_pixels),
                y0=int(self.y0 - extra_pixels),
                y1=int(self.y1 + extra_pixels),
            ).__bring_back_in(imshape).clip(imshape)

    def square(self, imshape: Tuple[int, int]):
        s = max(self.height, self.width)
        return Rect(
            x0=self.center.x - s // 2,
            x1=self.center.x + s // 2 + s % 2,
            y0=self.center.y - s // 2,
            y1=self.center.y + s // 2 + s % 2,
        ).__bring_back_in(imshape)

    def __bring_back_in(self, imshape: Tuple[int, int]):
        height, width = imshape[0], imshape[1]
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        if x0 < 0:
            x1 += -x0
            x0 = 0
        elif x1 >= width:
            x0 -= ((x1 - width) + 1)
            x1 = width - 1
        if y0 < 0:
            y1 += -y0
            y0 = 0
        elif y1 >= height:
            y0 -= ((y1 - height) + 1)
            y1 = height - 1
        return Rect(x0=x0, x1=x1, y0=y0, y1=y1)

    def clip(self, imshape: Tuple[int, int]):
        height, width = imshape[0], imshape[1]
        return Rect(
            x0=0 if self.x0 < 0 else self.x0,
            y0=0 if self.y0 < 0 else self.y0,
            x1=width - 1 if self.x1 >= width else self.x1,
            y1=height - 1 if self.y1 >= height else self.y1,
        )

def crop_from_rect(imshape: Tuple[int, int], crop_rect: Rect):
    whole_rect = Rect.from_points((0, 0), (imshape[1] - 1, imshape[0] - 1))
    xleft = crop_rect.topleft.x - whole_rect.topleft.x
    xright = whole_rect.bottomright.x - crop_rect.bottomright.x
    ytop = crop_rect.topleft.y - whole_rect.topleft.y
    ybot = whole_rect.bottomright.y - crop_rect.bottomright.y
    return (ytop, ybot), (xleft, xright), (0, 0)

def to_tuple(point: NpPosition, cast_to_int=True) -> Position:
    p = point.astype(int) if cast_to_int else point
    return Position(x=p[0].item(), y=p[1].item())

def to_tuple_flip(point: NpPosition, cast_to_int=True):
    p = to_tuple(point, cast_to_int)
    fp = flip_pair(p)
    return Position(x=fp[0], y=fp[1])

T = TypeVar('T')
U = TypeVar('U')

def flip_pair(t: Tuple[T, U]) -> Tuple[U, T]:
    """Flips a 2-ple"""
    return t[1], t[0]

def to_json(thing: Union[Dict, List]) -> str:
    return prettyjson(thing, maxlinelength=120)

def unzip(it):
    return map(list, zip(*it))

def eq_gen_it(gen_or_it_1: Union[Iterable, Generator], gen_or_it2: Union[Iterable, Generator]):
    # https://stackoverflow.com/a/9983596
    from itertools import zip_longest
    return all(a == b for a, b in zip_longest(gen_or_it_1, gen_or_it2, fillvalue=object()))

def rgb2gray(image: ColorImage) -> GrayscaleImage:
    from skimage.color import rgb2gray
    return (255 * rgb2gray(image)).astype('uint8')

class ProgressBar:
    def __init__(self, length, width=40):
        self.toolbar_width = width
        self.length = length
        self.progress_idx = 0
        # setup toolbar
        self.progressMsg = "Procesando 0/%d" % self.length
        sys.stdout.write(self.progressMsg + "[%s]" % (" " * self.toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (len(self.progressMsg) + self.toolbar_width + 2))  # return to start of line, after '['

    def next(self):
        self.progress_idx += 1
        progress = self.toolbar_width * self.progress_idx // self.length + 1
        progress_msg = "Procesando %d/%d " % (self.progress_idx, self.length)
        sys.stdout.write(progress_msg)
        sys.stdout.write("[%s%s]" % ("-" * progress, " " * (self.toolbar_width - progress)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (len(progress_msg) + self.toolbar_width + 2))  # return to start of line, after '['

def ensure_path(path: Union[Path, str]):
    if isinstance(path, str): path = Path(path)
    return path

def filehash(path: Union[Path, str]):
    import hashlib
    path = ensure_path(path)
    with path.open('rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    return h

class SerializableEnum(str, Enum):
    def _generate_next_value_(self, start, count, last_values):
        return self

class Colors(NamedTuple):
    BLACK = 0, 0, 0
    WHITE = 255, 255, 255
    GRAY = 185, 185, 185
    RED = 255, 0, 0
    GREEN = 0, 255, 0
    BLUE = 0, 0, 255

def draw_line(image: Image_T, start: Position, end: Position, color: Color = Colors.BLACK, width=1):
    from PIL import Image, ImageDraw
    pilimage = Image.fromarray(image.copy())
    draw = ImageDraw.ImageDraw(pilimage)
    draw.line([start, end], width=width, fill=color)
    return np.array(pilimage)

def draw_text(image: Image_T, text: Any, pos: Position, size=20, color: Color = Colors.BLACK):
    """"Returns a copy of `image` with `text` drawn on `pos`, with optional `size`"""
    from PIL import Image, ImageDraw, ImageFont
    text = str(text)
    font = ImageFont.truetype("arial.ttf", size=size)
    pilimage = Image.fromarray(image)
    draw = ImageDraw.ImageDraw(pilimage)
    draw.text((pos.x - size // 2, pos.y - size // 2), text, font=font, fill=color)
    return np.asarray(pilimage)

PixelOrPixels = TypeVar('PixelOrPixels', Pixel, Sequence[Pixel])

def blend(base: PixelOrPixels, top: Union[Color, Pixel], alpha) -> PixelOrPixels:
    if isinstance(top, tuple):
        top = np.array(top)
    return (base * (1 - alpha) + top * alpha).astype(int)

class Side(SerializableEnum):
    Bottom = auto()
    Top = auto()
    Left = auto()
    Right = auto()
    Center = auto()

    @staticmethod
    def center_rect(imshape: Tuple[int, int], percentage: float):
        smaller_dim = min(imshape[0], imshape[1])
        top = int(smaller_dim * percentage)
        bottom = int(imshape[0] - top)
        left = int(smaller_dim * percentage)
        right = int(imshape[1] - left)
        return Rect.from_points((left, top), (right, bottom))

    @staticmethod
    def from_point(point: Position, imshape: Tuple[int, int], percentage: float):
        smaller_dim = min(imshape[0], imshape[1])
        top = smaller_dim * percentage
        bottom = imshape[0] - top
        left = smaller_dim * percentage
        right = imshape[1] - left
        if point.y < top: return Side.Top
        if point.y > bottom: return Side.Bottom
        if point.x < left: return Side.Left
        if point.x > right: return Side.Right
        return Side.Center

square_rect_test = partial(Rect.square, imshape=(20, 20))

# assert (square_rect_test(Rect(0, 10, 0, 10)) == Rect(0, 10, 0, 10))
# assert (square_rect_test(Rect(0, 10, 5, 15)) == Rect(0, 10, 5, 15))
# assert (square_rect_test(Rect(0, 10, 0, 15)) == Rect(0, 15, 0, 15))
# assert (square_rect_test(Rect(0, 10, 10, 15)) == Rect(0, 10, 7, 17))
# assert (square_rect_test(Rect(0, 10, 16, 21)) == Rect(0, 10, 9, 19))
# assert (square_rect_test(Rect(0, 10, 16, 21)) == Rect(0, 10, 9, 19))
# assert (square_rect_test(Rect(-10, 5, 18, 22)) == Rect(0, 15, 4, 19))

def encode_np_randomstate(state):
    return state[0], state[1].tolist(), state[2], state[3], state[4]

def decode_np_randomstate(state):
    return state[0], np.array(state[1], dtype='uint32'), state[2], state[3], state[4]
