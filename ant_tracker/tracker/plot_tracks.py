import glob

import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from matplotlib.cm import get_cmap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .ant_labeler_info import LabelingInfo
from .common import unzip
from .info import TracksInfo
from .track import Loaded

plt.close('all')

def get_color(x, y, shape, xcolor='Spectral', ycolor='gray'):
    xcolormap = get_cmap(xcolor)
    ycolormap = get_cmap(ycolor)

    if np.isscalar(x) and np.isscalar(y):
        if type(x) == int: x = x / shape[1]
        if type(y) == int: y = y / shape[0]
        xc = np.array(xcolormap(x))
        yc = np.array(ycolormap(y))
    else:
        if x.dtype == int: x = x / shape[1]
        if x.dtype == int: y = y / shape[0]
        xc = xcolormap(x)
        yc = ycolormap(y)
    r = (xc + yc) / 2
    return r

def plot_tracks(info: TracksInfo):
    fig = plt.figure()
    is_labeled = isinstance(info, LabelingInfo)
    ax = plt.gca()
    rects = []
    leaves = []
    leaf_probs = []
    rect_h = 0.9
    rect_w = 1
    for itrack, track in enumerate(info.tracks):
        path = track.path()
        for x, y, frame in path:
            rects.append(
                Rectangle((frame, itrack - 0.5), rect_w, rect_h, color=get_color(int(x), int(y), info.video_shape),
                          )
            )
        if is_labeled:
            if track.loaded == Loaded.Yes:
                leaves.append((
                    (track.last_frame() + track.first_frame()) / 2,
                    itrack,
                ))
        else:
            leaf_probs.append((
                (track.last_frame() + track.first_frame()) / 2,
                itrack,
                track.load_probability,
            ))
    pc = PatchCollection(rects, match_original=True)
    ax.add_collection(pc)
    if is_labeled:
        if len(leaves) > 0:
            lx, ly = unzip(leaves)
            ax.scatter(lx, ly, marker="*", ec='k', s=60)
    else:
        for lx, ly, prob in leaf_probs:
            ax.text(lx, ly, f"{int(prob * 100)}%")
            ax.scatter(lx, ly, marker="*", ec='k', s=60, alpha=prob)
    ax.set_ylim(-0.5, len(info.tracks))
    ax.set_xlim(0, info.video_length)
    ax.set_ylabel("Track")
    ax.set_xlabel("Frame")
    ax.set_title(info.video_name + (" - GT" if is_labeled else ""))
    plt.show()

def __main():
    video = "HD1"
    version = "dev1"

    filename = f"data/{video}-2.0.2.{version}.trk"

    info = TracksInfo.load(filename)
    info_gt = LabelingInfo.load(glob.glob(f"vid_tags/**/{video}.tag")[0])

    shape = info.video_shape
    get_colors = partial(get_color, shape=shape)
    X, Y = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))
    Z = get_colors(X, Y)

    plt.figure()
    plt.imshow(Z)

    plot_tracks(info)
    plot_tracks(info_gt)

if __name__ == '__main__':
    __main()
