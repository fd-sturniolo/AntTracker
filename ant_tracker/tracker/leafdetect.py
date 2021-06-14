import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Tuple

from .blob import Blob
from .common import Video, ColorImage
from .track import Track

def _get_blob_rect(blob: Blob, imshape: Tuple[int, int], extra_pixels: int, square: bool):
    x0, x1, y0, y1 = blob.bbox.xxyy
    if y1 - y0 <= 1 or x1 - x0 <= 1: return None
    rect = blob.bbox
    if square:
        rect = (
            blob.bbox
                .scale(imshape, extra_pixels=extra_pixels)
                .square(imshape)
        )
    return rect

def _get_frame_slice(video: Video, blob: Blob, frame_n: int, extra_pixels: int, square=False) -> Optional[ColorImage]:
    image = video[frame_n]
    rect = _get_blob_rect(blob, image.shape, extra_pixels, square)
    if rect is None: return None
    x0, x1, y0, y1 = _get_blob_rect(blob, image.shape, extra_pixels, square).xxyy
    return image[y0:y1, x0:x1, :]

class LeafDetector(ABC):
    def __init__(self, video: Video, extra_pixels=15, img_size=64, n_frames=75):
        """
        Args:
            extra_pixels: número de píxeles alrededor del Blob que se toman para el slice de video
            img_size: tamaño del input al detector (img_size,img_size,3)
            n_frames: número de frames usado por el detector, los tracks se samplearán según sea necesario
        """
        self.video = video
        self.img_size = img_size
        self.n_frames = n_frames
        self.extra_pixels = extra_pixels

    def probability(self, track: Track) -> float:
        _input = self._track2input(track)
        if _input is None:
            return 0
        return self.call_model(_input)

    @abstractmethod
    def call_model(self, model_input):
        pass

    def _track2input(self, track: Track):
        from skimage.transform import resize
        blobs = track.get_safe_blobs(percentage=0.05)
        if len(blobs) == 0:
            blobs = track.get_safe_blobs(percentage=0.025)
            if len(blobs) == 0:
                return None
        slices = []
        for frame_n, blob in sorted(blobs.items()):
            slice_with_ant = _get_frame_slice(
                self.video,
                blob,
                frame_n,
                extra_pixels=self.extra_pixels,
                square=True
            )
            if slice_with_ant is None:
                continue
            slices.append(slice_with_ant)
        if len(slices) == 0: return None
        slices = np.array(slices)
        images = []
        indexes = np.round(np.linspace(0, len(slices) - 1, self.n_frames)).astype(int)
        for s in slices[indexes]:
            images.append(resize(s, (64, 64), preserve_range=True))
        images = np.stack(images, axis=0) / 255.0
        images = np.expand_dims(images, axis=0)
        return images

# noinspection PyUnresolvedReferences
class TFLeafDetector(LeafDetector):
    def __init__(self, model_folder: Union[Path, str], video: Video, extra_pixels=15, img_size=64, n_frames=75):
        import tensorflow as tf
        from .tf_model_reqs import F1Score
        self.model = tf.keras.models.load_model(model_folder, custom_objects={'F1Score': F1Score})
        super(TFLeafDetector, self).__init__(video, extra_pixels, img_size, n_frames)

    def call_model(self, model_input):
        return self.model(model_input).numpy()[0, 0]

# noinspection PyUnresolvedReferences
class TFLiteLeafDetector(LeafDetector):
    def __init__(self, tfl_model: Union[Path, str], video: Video, extra_pixels=15, img_size=64, n_frames=75):
        import tflite_runtime.interpreter as tflite
        self.interpreter = tflite.Interpreter(tfl_model)
        self.interpreter.allocate_tensors()
        self.in_index = self.interpreter.get_input_details()[0]['index']
        self.out_index = self.interpreter.get_output_details()[0]['index']
        super(TFLiteLeafDetector, self).__init__(video, extra_pixels, img_size, n_frames)

    def call_model(self, model_input):
        self.interpreter.set_tensor(self.in_index, model_input.astype(np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out_index).item()
