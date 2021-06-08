import os, sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import pims

from ant_tracker.tracker.segmenter import LogWSegmenter

testjson = "test_segmenter.json"
videofile = Path(ROOT_DIR)/"ant_tracker/tracker/vid_tags/HD9.mp4"

video = pims.PyAVReaderIndexed(videofile)
seg = LogWSegmenter(video)

for frame_n, blobs in seg.frames_with_blobs:
    if frame_n == 3:
        break
with open(testjson, 'w') as f:
    f.write(seg.serialize())

seg2 = LogWSegmenter.deserialize(filename=testjson)
os.remove(testjson)
seg2.set_video(video)

for frame_n, blobs in seg2.segment_rolling_continue():
    if frame_n == 6:
        break
for frame_n, blobs in seg.segment_rolling_continue():
    if frame_n == 6:
        break

assert seg.serialize() == seg2.serialize()
print("Test passed")
