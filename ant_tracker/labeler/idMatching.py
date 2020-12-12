from classes import Tracker, TrackedAnt, TrackingState, getNextColor, first
import cv2 as cv
from typing import List
import logging

# logging.basicConfig(filename="idMatch.log",format="[%(asctime)s]: %(message)s",filemode="w",level=logging.DEBUG)

# def draw_ants(img, ants: List[TrackedAnt], frame: int):
#     for ant in ants:
#         color = getNextColor.kelly_colors[ant.id%len(getNextColor.kelly_colors)]
#         rect: Rect = first(ant.rects, lambda r: r.frame == frame)
#         if rect is not None:
#             x,y,w,h = rect.unpack()
#             img = cv.rectangle(img, (x,y), (x+w,y+h), color, 2)
#             img = cv.putText(img,str(ant.id),(x,y),cv.FONT_HERSHEY_SIMPLEX,1,255)
#             img = cv.putText(img,TrackingState.toString(ant.state,True),(x-10,y-3),cv.FONT_HERSHEY_SIMPLEX,1,255)
#     return img

# cv.namedWindow("tracked",cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)
# cv.namedWindow("labeled",cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

# labeled = Tracker.deserialize(filename="dia-labeled.rtg")
tracked = Tracker.deserialize(filename="dia-tracked.rtg")

for ant in tracked.getAntsThatDidntCross():
    print(ant.getVelocity())
    print(ant.getVelocityAtFrame(14))
    break

# tracked.modifyOwnIdsToMatch(labeled)

# video = cv.VideoCapture("dia.mp4")
# for frame in range(0,500):
#     _,originalFrame = video.read()
#     img = originalFrame.copy()
#     img = draw_ants(img,tracked.trackedAnts,frame)
#     cv.imshow("tracked",img)

#     imgColl = originalFrame.copy()
#     imgColl = draw_ants(imgColl,labeled.trackedAnts,frame)
#     cv.imshow("labeled",imgColl)
#     k = cv.waitKey(0) & 0xff
#     if k == 27:
#         break
