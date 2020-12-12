import pims

from .common import to_json
from .segmenter import LogWSegmenter
from .tracking import Tracker

testjson = "test_tracker.json"
videofile = "vid_tags/720x510/HD720_1.mp4"

video = pims.PyAVReaderIndexed(videofile)[:30]
tracker = Tracker(videofile, LogWSegmenter(video))
for frame_n in tracker.track_progressive():
    if frame_n == 6:
        break
tracker.save_unfinished(testjson)

tracker2 = Tracker.load_unfinished(testjson)
for frame_n in tracker2.track_progressive_continue():
    if frame_n == 9:
        break

for frame_n in tracker.track_progressive_continue():
    if frame_n == 9:
        break

t1e = tracker.encode_unfinished()
t2e = tracker2.encode_unfinished()

assert to_json(t1e) == to_json(t2e)
print("Test passed")
import os

os.remove(testjson)

# class Timer:
#     def __init__(self, title: str = ""):
#         self.title = title
#
#     def __enter__(self):
#         import timeit
#         print("-" * 6 + self.title + "-" * 6)
#         self.start = timeit.default_timer()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         import timeit
#         print("-" * 6 + f"{self.title} end. Time: ", timeit.default_timer() - self.start)
#
# tracker = Tracker(videofile, LogWSegmenter(video))
# with Timer("Running all at once"):
#     for frame_n in tracker.track_progressive():
#         pass
#
# tracker2 = Tracker(videofile, LogWSegmenter(video))
# with Timer("Stopping then continuing"):
#     for frame_n in tracker2.track_progressive():
#         if frame_n == 10:
#             break
#     for frame_n in tracker2.track_progressive_continue():
#         pass
