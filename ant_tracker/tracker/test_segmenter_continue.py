import pims

from .segmenter import LogWSegmenter

testjson = "test_segmenter.json"
videofile = "vid_tags/720x510/HD720_1.mp4"

video = pims.PyAVReaderIndexed(videofile)
seg = LogWSegmenter(video)

for frame_n, blobs in seg.frames_with_blobs:
    if frame_n == 3:
        break
with open(testjson, 'w') as f:
    f.write(seg.serialize())

seg2 = LogWSegmenter.deserialize(filename=testjson)
seg2.set_video(video)

for frame_n, blobs in seg2.segment_rolling_continue():
    if frame_n == 6:
        break
for frame_n, blobs in seg.segment_rolling_continue():
    if frame_n == 6:
        break

assert seg.serialize() == seg2.serialize()
print("Test passed")
import os

os.remove(testjson)
