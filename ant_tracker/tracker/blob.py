import numpy as np
from memoized_property import memoized_property
from typing import List, Optional, Tuple, TypedDict

from .common import BinaryMask, Color, ColorImage, Contour, Image_T, NpPosition, Position, Rect, Side, to_array, \
    to_tuple, to_tuple_flip, Colors, GrayscaleImage
from .kellycolors import KellyColors

class Blob:
    def __init__(self, *, imshape: Tuple[int, int], mask: BinaryMask = None, contour: Contour = None,
                 approx_tolerance=1):
        if (mask is None) == (contour is None):
            raise ValueError("Only mask or contour, not both")
        if mask is not None:
            import cv2 as cv
            contour = cv.findContours(mask.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)[0][0]
            self.contour = np.flip(np.squeeze(cv.approxPolyDP(contour, approx_tolerance, True)), axis=1)
        elif contour is not None:
            self.contour = contour
        self.contour = _clip_contour(self.contour, imshape)
        self.center = self.contour.mean(axis=0)
        self.shape = imshape

    def __repr__(self):
        return f"Blob(at={self.center_xy})"

    @property
    def bbox(self) -> Rect:
        ymin = np.min(self.contour[:, 0])
        ymax = np.max(self.contour[:, 0])
        xmin = np.min(self.contour[:, 1])
        xmax = np.max(self.contour[:, 1])
        return Rect.from_points((xmin, ymin), (xmax, ymax))

    @property
    def center_xy(self):
        return to_tuple_flip(self.center)

    @memoized_property
    def area(self):
        p = self.props
        if not p: return 1
        return p.area

    def is_fully_visible(self, percentage):
        center_rect = Side.center_rect(self.shape, percentage)
        return (self.bbox.topleft in center_rect) and (self.bbox.bottomright in center_rect)

    @memoized_property
    def is_touching_border(self):
        for pixel in np.array(self.full_contour).T:
            if (
                    pixel[0] == self.shape[0] - 1 or
                    pixel[1] == self.shape[1] - 1 or
                    pixel[0] == 0 or
                    pixel[1] == 0
            ): return True
        return False

    @property
    def full_contour(self) -> Tuple[np.ndarray, np.ndarray]:
        from skimage.draw import line, polygon_perimeter
        if len(self.contour[:, 0]) == 1:
            # moments_coords_central needs at least 2 points
            return (
                np.array([self.contour[0, 0], self.contour[0, 0]]),
                np.array([self.contour[0, 1], self.contour[0, 1]])
            )
        if len(self.contour[:, 0]) == 2:
            return line(self.contour[0, 0], self.contour[0, 1], self.contour[1, 0], self.contour[1, 1])
        try:
            return polygon_perimeter(self.contour[:, 0], self.contour[:, 1], shape=self.shape, clip=True)
        except IndexError:
            # sometimes, when near the border, polygon_perimeter fails. I tried to make it always work, but no dice
            # so just take the original contour and get out
            return self.contour[:, 0], self.contour[:, 1]

    @memoized_property
    def radius(self):
        return np.linalg.norm(self.contour - self.center, axis=1).max(initial=0)

    @memoized_property
    def props(self):
        from skimage.measure import regionprops
        mask = self.get_mask()
        r = regionprops(mask.astype(np.uint8))
        if len(r) == 0:  # flat or near-flat blob
            return None
        return r[0]

    @property
    def length(self):
        p = self.props
        if not p: return 1
        return p.major_axis_length

    @property
    def width(self):
        p = self.props
        if not p: return 1
        return p.minor_axis_length

    def get_mask(self):
        from skimage.draw import polygon
        fc = self.full_contour
        rr, cc = polygon(fc[0], fc[1], self.shape)
        ret = np.zeros(self.shape, dtype=bool)
        ret[rr, cc] = True
        return ret

    # region Creation

    def new_moved_to(self, to: NpPosition, imshape: Tuple[int, int]):
        contour = _clip_contour(np.round(self.contour + (to.flatten() - self.center)), imshape)

        # check for flat blobs: most likely moved too far out
        ymax = contour[:, 0].max()
        ymin = contour[:, 0].min()
        xmax = contour[:, 1].max()
        xmin = contour[:, 1].min()

        if ymax == ymin:  # all y are equal
            if ymax == self.shape[0] - 1:  # and equal to the max possible
                contour[-1, 0] = ymax - 1  # make a single one different so the blob isn't flat
            else:  # also takes care of cases that aren't on the border
                contour[:-1, 0] = ymin + 1
        if xmax == xmin:
            if xmax == self.shape[1] - 1:
                contour[:-1, 1] = xmax - 1
            else:
                contour[:-1, 1] = xmin + 1
        return Blob(contour=contour, imshape=imshape)

    # endregion
    # region Drawing

    # noinspection PyTypeChecker
    def draw_contour(self, img: ColorImage, *, frame_id: int = None,
                     color: Optional[Color] = None, alpha=0.5,
                     filled: bool = True,
                     text: str = None, text_color: Color = Colors.BLACK) -> ColorImage:
        import skimage.draw as skdraw
        from .common import blend
        """Returns a copy of `img` with `self.contour` drawn"""
        if frame_id is not None:
            color = KellyColors.get(frame_id)
            text = str(frame_id)
        else:
            color = color if color is not None else KellyColors.next()

        copy: ColorImage = img.copy()
        fc = self.full_contour
        if np.any(fc):
            copy[fc[0], fc[1]] = blend(copy[fc[0], fc[1]], color, 1)
            if filled:
                rr, cc = skdraw.polygon(fc[0], fc[1], shape=self.shape)
                copy[rr, cc] = blend(copy[rr, cc], color, alpha)
            if text is not None:
                copy = self.draw_label(copy, text=text, color=text_color)
        return copy

    def draw_label(self, image: Image_T, text: str, *, color: Color = Colors.BLACK, size=20, separation=9):
        from .common import draw_text
        import skimage.draw as skdraw
        copy = image.copy()
        vector_to_center_of_img = np.array(copy.shape[0:2]) / 2 - self.center
        norm = np.linalg.norm(vector_to_center_of_img)
        magnitude = np.log1p(norm) * separation
        vector_to_center_of_img = vector_to_center_of_img * magnitude / norm

        pos = to_tuple_flip(self.center + vector_to_center_of_img)

        rr, cc = skdraw.line(self.center_xy.y, self.center_xy.x, pos.y, pos.x)
        rr, cc = np.clip(rr, 0, self.shape[0]), np.clip(cc, 0, self.shape[1])
        copy[rr, cc] = (20, 20, 20)
        copy = draw_text(copy, text=text, size=size, pos=pos, color=color)
        return copy

    @classmethod
    def draw_blobs(cls, blobs: List['Blob'], image: ColorImage) -> ColorImage:
        """Returns a copy of `image` with all `blobs` drawn onto it, with its index as a label"""
        copy: ColorImage = image.copy()
        for i, blob in enumerate(blobs):
            copy = blob.draw_contour(copy, frame_id=i)
        return copy

    @classmethod
    def make_label_image(cls, blobs: List['Blob'], imshape: Tuple[int, int]) -> GrayscaleImage:
        import skimage.draw as skdraw
        img = np.zeros(imshape, dtype='int')
        for color, blob in enumerate(blobs, start=1):
            fc = blob.full_contour
            if np.any(fc):
                rr, cc = skdraw.polygon(fc[0], fc[1], shape=imshape)
                img[rr, cc] = color
        return img

    # endregion
    # region Serialization

    class Serial(TypedDict):
        contour: List[Position]

    def encode(self) -> 'Blob.Serial':
        return {'contour': [to_tuple(point) for point in self.contour]}

    @classmethod
    def decode(cls, ant_as_dict: 'Blob.Serial', imshape: Tuple[int, int]) -> 'Blob':
        contour = _clip_contour(np.array([to_array(point) for point in ant_as_dict["contour"]]), imshape)
        return cls(contour=contour, imshape=imshape)

    # endregion

def _clip_contour(contour, imshape):
    contour[:, 0] = np.clip(contour[:, 0], 0, imshape[0] - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, imshape[1] - 1)
    return contour.astype('int32')
