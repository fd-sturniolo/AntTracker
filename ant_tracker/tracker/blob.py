import numpy as np
from memoized_property import memoized_property
from typing import List, Optional, Tuple, TypedDict

from .common import BinaryMask, Color, ColorImage, Contour, Image_T, NpPosition, Position, Rect, Side, \
    to_tuple, to_tuple_flip, Colors, GrayscaleImage
from .kellycolors import KellyColors

class Blob:
    def __init__(self, *, imshape: Tuple[int, int], mask: BinaryMask = None, contour: Contour = None,
                 approx_tolerance=1):
        if (mask is None) == (contour is None):
            raise ValueError("Only mask or contour, not both")
        if mask is not None:
            import cv2 as cv
            contours = cv.findContours(mask.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)[0]
            #! Puede venir más de un contorno, por segmenter.py:L122:135
            c = max(contours, key = cv.contourArea)
            self.contour = np.flip(np.squeeze(cv.approxPolyDP(c, approx_tolerance, True), axis=1), axis=1)
            self.contour = _clip_contour(self.contour, imshape)
        elif contour is not None:
            self.contour = contour
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

    @property
    def area(self):
        try:
            return self._area
        except AttributeError:
            self.load_props()
            return self._area

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
        from skimage.draw import polygon_perimeter
        try:
            return polygon_perimeter(self.contour[:, 0], self.contour[:, 1], shape=self.shape, clip=True)
        except IndexError:
            # sometimes, when near the border, polygon_perimeter fails.
            # also when there is only a single pixel in `self.contour`

            #! in <v1.0.9 contour might be out of bounds
            #TODO: Issue #14 should fix this at the info decoding level
            c = _clip_contour(self.contour, self.shape)
            return c[:, 0], c[:, 1]

    @memoized_property
    def radius(self):
        return np.linalg.norm(self.contour - self.center, axis=1).max(initial=0) or 1

    def load_props(self):
        from skimage.measure import regionprops
        mask = self.get_mask()
        r = regionprops(mask.astype(np.uint8), cache=True)
        if len(r) == 0:  # flat or near-flat blob
            self._length = 1
            self._width = 1
            self._area = 1
        else:
            self._length = r[0].major_axis_length or 1
            self._width = r[0].minor_axis_length or 1
            self._area = r[0].area or 1
        del r

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
        try:
            return self._length
        except AttributeError:
            self.load_props()
            return self._length

    @property
    def width(self):
        try:
            return self._width
        except AttributeError:
            self.load_props()
            return self._width

    def get_mask(self):
        from skimage.draw import polygon, line

        ret = np.zeros(self.shape, dtype=bool)
        fc = self.full_contour
        # vertices
        ret[fc[0], fc[1]] = True

        # edges
        if len(fc[0]) > 1:
            rr, cc = polygon(fc[0], fc[1], self.shape)
            if not len(rr):
                rr, cc = line(fc[0][0], fc[1][0], fc[0][1], fc[1][1])
                rr = np.clip(rr, 0, self.shape[0] - 1)
                cc = np.clip(cc, 0, self.shape[1] - 1)
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
        """Returns a copy of `img` with `self.contour` drawn"""
        from .common import blend
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
                mask = self.get_mask()
                copy[mask] = blend(copy[mask], color, alpha)
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
        rr = np.clip(rr, 0, self.shape[0] - 1)
        cc = np.clip(cc, 0, self.shape[1] - 1)
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
        img = np.zeros(imshape, dtype='int')
        for color, blob in enumerate(blobs, start=1):
            mask = blob.get_mask()
            img[mask] = color
        return img

    # endregion
    # region Serialization

    class Serial(TypedDict):
        contour: List[Position]

    def encode(self) -> 'Blob.Serial':
        return {'contour': [to_tuple(point) for point in self.contour]}

    @classmethod
    def decode(cls, ant_as_dict: 'Blob.Serial', imshape: Tuple[int, int]) -> 'Blob':
        return cls(contour=np.array(ant_as_dict["contour"]), imshape=imshape)

    # endregion

def _clip_contour(contour, imshape):
    contour[:, 0] = np.clip(contour[:, 0], 0, imshape[0] - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, imshape[1] - 1)
    return contour.astype('int32')
