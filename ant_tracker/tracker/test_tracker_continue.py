import os, sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class Timer:
    def __init__(self, title: str = ""):
        self.title = title

    def __enter__(self):
        import timeit
        print("-" * 6 + self.title + "-" * 6)
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        import timeit
        print("-" * 6 + f"{self.title} end. Time: ", timeit.default_timer() - self.start)

import pims

from ant_tracker.tracker.common import to_json
from ant_tracker.tracker.segmenter import LogWSegmenter
from ant_tracker.tracker.tracking import Tracker

closed = ".testclosed.uctrk"
ongoing = ".testongoing.uotrk"
try: os.remove(closed)
except: pass
try: os.remove(ongoing)
except: pass
videofile = Path(ROOT_DIR)/"ant_tracker/tracker/vid_tags/HD9.mp4"

n_frames = 50
xtra_frames = 25
video = pims.PyAVReaderIndexed(videofile)[:n_frames+xtra_frames]

tracker = Tracker(videofile, LogWSegmenter(video))
for frame_n in tracker.track_progressive():
    print(f"\rTracking at frame {frame_n}", end="", flush=True)
    if frame_n == n_frames:
        break
print()
with Timer(f"Saving, {n_frames} frames"):
    tracker.save_unfinished(closed, ongoing)

with Timer(f"Loading, {n_frames} frames"):
    tracker2 = Tracker.load_unfinished(closed, ongoing, video, videofile)
for frame_n in tracker2.track_progressive_continue():
    print(f"\rTracking2 at frame {frame_n}", end="", flush=True)
    if frame_n == n_frames + xtra_frames:
        break
print()

for frame_n in tracker.track_progressive_continue():
    print(f"\rTracking at frame {frame_n}", end="", flush=True)
    if frame_n == n_frames + xtra_frames:
        break
print()

tracks = [track.encode() for track in tracker._Tracker__inprogress_tracks]
tracks2 = [track.encode() for track in tracker2._Tracker__inprogress_tracks]

print(f"tracks2 closed: {len(tracker2.closed(tracks2))}, ongoing: {len(tracker2.ongoing(tracks2))}")
print(f"tracks closed: {len(tracker.closed(tracks))}, ongoing: {len(tracker.ongoing(tracks))}")

assert to_json(tracks) == to_json(tracks2)
print("Test passed")

os.remove(closed)
os.remove(ongoing)

#
# tracker = Tracker(videofile, LogWSegmenter(video))
# with Timer("Running all at once"):
#     for frame_n in tracker.track_progressive():
#         if frame_n == 100:
#             break
#
# tracker2 = Tracker(videofile, LogWSegmenter(video))
# with Timer("Stopping then continuing"):
#     for frame_n in tracker2.track_progressive():
#         if frame_n == 50:
#             break
#     for frame_n in tracker2.track_progressive_continue():
#         pass
