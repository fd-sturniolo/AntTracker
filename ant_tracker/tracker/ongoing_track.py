from dataclasses import dataclass, field

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Optional, Tuple, List, Dict, TypedDict, Union

from .blob import Blob
from .common import ColorImage, FrameNumber, Vector, to_tuple_flip, Side
from .track import Track

SAMPLE_RATE = 1  # seconds/frame

def kalman_filter_init(sample_rate=1):
    kf = KalmanFilter(dim_z=2, dim_x=4)
    dt = sample_rate
    # Measurement noise (pretty low, b/c segmentation is fairly high fidelity)
    kf.R = np.array([[1, 0],
                     [0, 1]]) * .01
    # Process noise (from Taylor expansion?)
    kf.Q = np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                     [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                     [dt ** 3 / 2, 0, dt ** 2, 0],
                     [0, dt ** 3 / 2, 0, dt ** 2]]) * 10
    # Initial estimate variance
    kf.P = kf.Q.copy()
    # State update/transition matrix
    kf.F = np.array([[1, 0, dt, 0],  # x = x_0 + dt*v_x
                     [0, 1, 0, dt],  # y = y_0 + dt*v_y
                     [0, 0, 1, 0],  # v_x = v_x0
                     [0, 0, 0, 1]])  # v_y = v_y0
    # Control matrix (acceleration)
    kf.B = np.array([[dt ** 2 / 2, 0],  # + dt^2*a_x/2
                     [0, dt ** 2 / 2],  # + dt^2*a_y/2
                     [dt, 0],  # + dt*a_x
                     [0, dt]])  # + dt*a_y
    # Measurement matrix (measure only x & y)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    return kf

@dataclass
class OngoingTrack(Track):
    kf: KalmanFilter = field(init=False, default_factory=lambda: kalman_filter_init(sample_rate=SAMPLE_RATE),
                             repr=False)
    a_sigma: float = field(repr=False)
    frames_lost: int = field(init=False, default=0)
    closed: bool = field(init=False, default=False)

    imshape: Tuple[int, int] = field(repr=False)
    frames_until_close: int = field(repr=False)
    can_predict: bool = field(init=False, default=True)
    random_state: np.random.RandomState

    def __post_init__(self):
        blob = self.blobs[min(self.blobs.keys())]
        self.kf.x = np.reshape(np.hstack((blob.center, [0, 0])), (4, 1))

    def __repr__(self):
        s = f"OTrack(T{self.id}, x={self.x}"
        if self.closed:
            s += ", closed"
        if self.is_currently_lost:
            s += f", frames_lost: {self.frames_lost}/{self.frames_until_close}"
        s += ")"
        return s

    @property
    def x(self):
        return self.kf.x.flatten()

    @property
    def is_currently_lost(self):
        return self.frames_lost > 0

    @property
    def is_exiting(self):
        if Side.from_point(self.last_blob().center_xy, self.imshape, percentage=0.05) is Side.Center:
            return False
        return True

    def as_closed(self) -> Track:
        return Track(self.id, self.blobs)

    def predict(self, u: Optional[Vector] = None):
        if not self.can_predict:
            raise ValueError(f'Cannot predict on {self}. update() first')
        if self.closed:
            raise ValueError(f'Cannot predict on {self}. Track closed')
        u_ = self.random_state.rand(2, 1)
        self.kf.predict(u=u_ * self.a_sigma)
        self.can_predict = False
        return self.x

    def update(self, frame: FrameNumber, blob: Optional[Blob] = None):
        """
        If track wasn't found in a frame, ``blob`` should be ``None``, to update using last prediction.

        Closes the track and does nothing if ``frames_until_close`` was reached
        """
        if self.closed:
            raise ValueError(f'Cannot update {self} because it\'s closed.'
                             ' Call as_closed() and replace the track,'
                             ' or filter by (not closed)')
        if blob is None:
            if self.frames_lost > self.frames_until_close or self.is_exiting:
                self.close()
                return
            # lower velocity estimate, probably a collision
            self.kf.x[2:4] = self.kf.x[2:4] * self.a_sigma
            self.kf.update((self.kf.x[0:2] - self.kf.x[2:4]).T)
            newblob = self.at(frame - 1).new_moved_to(self.kf.x[0:2, :], self.imshape)
            self.__add_blob(frame, newblob)
            self.frames_lost += 1
        else:
            self.frames_lost = 0
            self.kf.update(blob.center)
            self.__add_blob(frame, blob)
        self.can_predict = True

    def close(self):
        for frame in range(self.last_frame() - self.frames_lost + 1, self.last_frame() + 1):
            self.blobs.pop(frame)
        self.closed = True
        self.can_predict = False

    def __add_blob(self, frame: FrameNumber, blob: Blob):
        last_frame = max(self.blobs.keys())
        if frame != last_frame + 1:
            raise ValueError(f'Tried to add {blob} to frame {frame}, where frame should\'ve been {last_frame + 1}. \n'
                             f'Frames: {list(self.blobs.keys())}')
        self.blobs[frame] = blob

    def draw_blob(self, frame: FrameNumber, image: ColorImage, unused_param=None) -> ColorImage:
        from .common import draw_line, blend, Colors
        import skimage.draw as skdraw
        copy = Track.draw_blob(self, frame, image, Colors.GRAY if self.is_currently_lost else None)
        center = self.at(frame).center_xy
        predicted_next_pos = to_tuple_flip(self.x[2:4] + self.at(frame).center)
        copy = draw_line(copy, center, predicted_next_pos)
        rr, cc = skdraw.circle_perimeter(predicted_next_pos.y, predicted_next_pos.x, 2, shape=image.shape)
        copy[rr, cc] = blend(copy[rr, cc], Colors.BLACK, 0.8)
        if self.is_currently_lost and not self.closed:
            copy = self.draw_x(frame, copy)
            return self.last_blob().draw_label(copy, text=f'since: {self.frames_lost}', size=10, separation=7)
        else:
            return copy

    def draw_x(self, frame: FrameNumber, image: ColorImage) -> ColorImage:
        from .common import Colors

        def cross(x, y, shape, size=2):
            import skimage.draw as skdraw
            lf = x - size if x - size >= 0 else 0
            rg = x + size if x + size < shape[1] else shape[1] - 1
            tp = y - size if y - size >= 0 else 0
            bt = y + size if y + size < shape[0] else shape[0] - 1
            rr1, cc1 = skdraw.line(tp, lf, bt, rg)
            rr2, cc2 = skdraw.line(bt, lf, tp, rg)
            return np.hstack((rr1, rr2)), np.hstack((cc1, cc2))

        center = self.at(frame).center_xy
        rr, cc = cross(center.x, center.y, image.shape)
        copy = image.copy()
        copy[rr, cc] = Colors.RED
        return copy

    # region Invalid overrides

    @property
    def loaded(self):
        """Do not use, get a :class:`Track` instance first with :method:`as_closed`"""
        raise AttributeError

    # endregion
    # region Serialization

    class Serial(TypedDict):
        id: int
        blobs: Dict[str, Blob.Serial]
        kf: Dict[str, Union[str, int]]
        frames_lost: int

    def encode(self) -> 'OngoingTrack.Serial':
        return {
            'id':          self.id,
            'blobs':       {str(i): blob.encode() for i, blob in self.blobs.items()},
            'kf':          encode_kalman(self.kf),
            'frames_lost': self.frames_lost,
        }

    # noinspection PyMethodOverriding
    @classmethod
    def decode(cls, track_serial: 'OngoingTrack.Serial',
               a_sigma: float, imshape: Tuple[int, int], frames_until_close: int,
               random_state=np.random.RandomState) -> 'OngoingTrack':

        self = cls(id=track_serial['id'],
                   blobs={int(i): Blob.decode(blob, imshape) for i, blob in track_serial["blobs"].items()},
                   a_sigma=a_sigma, imshape=imshape, frames_until_close=frames_until_close, random_state=random_state)
        self.kf = decode_kalman(track_serial['kf'])
        self.frames_lost = track_serial['frames_lost']
        return self

    # endregion

def encode_numpy(array: np.ndarray):
    return array.tolist()

def decode_numpy(list_: List) -> np.ndarray:
    return np.array(list_)

def encode_kalman(kf: KalmanFilter):
    return {
        'dim_x':   kf.dim_x,
        'dim_z':   kf.dim_z,
        'dim_u':   kf.dim_u,
        'x':       encode_numpy(kf.x),
        'P':       encode_numpy(kf.P),
        'x_prior': encode_numpy(kf.x_prior),
        'P_prior': encode_numpy(kf.P_prior),
        'x_post':  encode_numpy(kf.x_post),
        'P_post':  encode_numpy(kf.P_post),
        'F':       encode_numpy(kf.F),
        'Q':       encode_numpy(kf.Q),
        'R':       encode_numpy(kf.R),
        'H':       encode_numpy(kf.H),
        'K':       encode_numpy(kf.K),
        'y':       encode_numpy(kf.y),
        'S':       encode_numpy(kf.S),
        'SI':      encode_numpy(kf.SI),
        'M':       encode_numpy(kf.M),
        'B':       encode_numpy(kf.B),
        'alpha':   kf.alpha,
    }

def decode_kalman(serial) -> KalmanFilter:
    kf = KalmanFilter(dim_x=serial['dim_x'], dim_z=serial['dim_z'], dim_u=serial['dim_u'])
    kf.x = decode_numpy(serial['x'])
    kf.P = decode_numpy(serial['P'])
    kf.x_prior = decode_numpy(serial['x_prior'])
    kf.P_prior = decode_numpy(serial['P_prior'])
    kf.x_post = decode_numpy(serial['x_post'])
    kf.P_post = decode_numpy(serial['P_post'])
    kf.F = decode_numpy(serial['F'])
    kf.Q = decode_numpy(serial['Q'])
    kf.R = decode_numpy(serial['R'])
    kf.H = decode_numpy(serial['H'])
    kf.K = decode_numpy(serial['K'])
    kf.y = decode_numpy(serial['y'])
    kf.S = decode_numpy(serial['S'])
    kf.SI = decode_numpy(serial['SI'])
    kf.M = decode_numpy(serial['M'])
    kf.B = decode_numpy(serial['B'])
    kf.alpha = serial['alpha']
    return kf
