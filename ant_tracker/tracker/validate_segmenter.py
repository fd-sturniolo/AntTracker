import glob

import matplotlib.pyplot as plt
import numpy as np
import pims
import skimage.metrics
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
from skimage.color import label2rgb
from typing import Tuple, List

from .ant_labeler_info import LabelingInfo
from .blob import Blob
from .common import Video
from .segmenter import DohSegmenter, LogWSegmenter, Blobs

file = "HD2"
ver = "2.0.2.dev1"
segClass = {"2.0.2.dev1": LogWSegmenter, "2.0.2.dev2": DohSegmenter}[ver]

segmenter = segClass.deserialize(filename=glob.glob(f"./testdata/segmenter/{file}*{ver}*")[0])

fblobs = list(segmenter.frames_with_blobs)

truth = LabelingInfo.load(glob.glob(f"vid_tags/**/{file}.tag")[0])

def blobs_scores(blobs_true: Blobs, blobs_pred: Blobs, imshape: Tuple[int, int]):
    if len(blobs_true) == 0 and len(blobs_pred) == 0: return 0, 1, 1
    im_true = Blob.make_label_image(blobs_true, imshape)
    im_pred = Blob.make_label_image(blobs_pred, imshape)
    _, precision, recall = skimage.metrics.adapted_rand_error(im_true, im_pred)
    recall_weighted_rand_error = 1 - (5 * precision * recall / (4 * precision + recall))

    return recall_weighted_rand_error, precision, recall

def draw_segmentation(video: Video, all_blobs_true: List[Blobs], all_blobs_pred: List[Blobs]):
    fig_comp: Figure = plt.figure(constrained_layout=False)
    fig_comp.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    gs = fig_comp.add_gridspec(2, 2, height_ratios=[1, 0.5], wspace=0.0, hspace=0.1)
    ax_true: Axes = fig_comp.add_subplot(gs[0, 0])
    ax_pred: Axes = fig_comp.add_subplot(gs[0, 1])
    # ax_distances: Axes = fig.add_subplot(gs[1, :])

    ax_true.set_title('Ground Truth')
    ax_pred.set_title('Segment')

    ax_frame_slider = fig_comp.add_axes([0.1, 0.05, 0.8, 0.04])
    frame_slider = Slider(ax_frame_slider, 'Frame', 0, len(video), 0, valfmt="%d", valstep=1)

    def __frame_update_fn(val):
        nonlocal frame_n
        frame_n = int(val)
        draw_figure()

    frame_slider.on_changed(__frame_update_fn)

    def on_key_press(event):
        nonlocal play, frame_slider
        if event.key == 'a':
            frame_slider.set_val((frame_slider.val - 1) % len(video))
        elif event.key == 'd':
            frame_slider.set_val((frame_slider.val + 1) % len(video))
        elif event.key == 'p':
            play = not play
        elif event.key == 'escape' or event.key == 'q':
            import sys
            sys.exit()

    fig_comp.canvas.mpl_connect('key_press_event', on_key_press)

    def draw_figure():
        from .plotcommon import Animate
        nonlocal frame_slider
        frame_slider.val = frame_n
        frame = video[frame_n]

        label_true = Blob.make_label_image(all_blobs_true[frame_n], frame.shape[:2])
        label_pred = Blob.make_label_image(all_blobs_pred[frame_n], frame.shape[:2])

        frame_true = label2rgb(label_true, frame, bg_label=0)
        frame_pred = label2rgb(label_pred, frame, bg_label=0)

        Animate.draw(ax_true, frame_true)
        Animate.draw(ax_pred, frame_pred)

        fig_comp.canvas.draw_idle()

    frame_n = 0
    exit_flag = False
    play = False
    last_drawn_frame_n = -1
    while not exit_flag and frame_n < len(video):
        if last_drawn_frame_n != frame_n or play:
            draw_figure()
            last_drawn_frame_n = frame_n
            if play:
                frame_n += 1
        plt.pause(0.001)

rand_errors = []
ps = []
rs = []
for frame in range(truth.video_length):
    blobs_true = truth.get_blobs_in_frame(frame)
    blobs_pred = fblobs[frame]
    rand_error, p, r = blobs_scores(blobs_true, blobs_pred, segmenter.video_shape)
    rand_errors.append(rand_error)
    ps.append(p)
    rs.append(r)

plt.style.use('seaborn-deep')

plt.figure()
plt.title(file)
plt.plot(rand_errors, c='r', label="1-F2")
plt.axhline(np.nanmean(rand_errors), ls='dashed', c='r')
plt.ylim(0, 1)
plt.ylabel("Adapted Rand F2 Error")
plt.xlabel("Frame")

ax2 = plt.gca().twinx()
ax2.plot(rs, c='g', label="Recall")
ax2.plot(ps, c='b', label="Precision")
ax2.axhline(np.nanmean(rs), ls='dashed', c='g')
ax2.axhline(np.nanmean(ps), ls='dashed', c='b')

number_of_blobs_per_frame = [len(truth.get_blobs_in_frame(frame)) for frame in range(truth.video_length)]
ax3 = plt.gca().twinx()
ax3.bar(list(range(truth.video_length)), number_of_blobs_per_frame, alpha=0.3)
ax3.set_ylabel('N° hormigas presentes')
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

results = {
    'segmenter': segmenter.serialize()
}
draw_segmentation(pims.PyAVReaderIndexed(glob.glob(f"vid_tags/**/{file}.mp4")[0]),
                  [truth.get_blobs_in_frame(frame) for frame in range(truth.video_length)],
                  fblobs)
