import sys
import cv2 as cv
import numpy as np
import json

cv.namedWindow("Máscaras",cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)
with open("./Video16cr.tag",'r') as file:
    antCollection = json.load(file)
    videoShape = antCollection["videoShape"]
    videoSize = antCollection["videoSize"]
    for unlabeledFrame in antCollection["unlabeledFrames"]:
        packed_mask = unlabeledFrame["packed_mask"]
        packed_mask_ndarray = np.array(packed_mask,dtype='uint8')
        # print(packed_mask_ndarray)
        mask = np.unpackbits(packed_mask_ndarray, axis=None)[:videoSize].reshape(videoShape).astype('uint8')*255
        # print(mask)
        cv.imshow("Máscaras",mask)
        cv.waitKey(1)

# cap = cv.VideoCapture("../Video16c.mp4")


cv.waitKey(0)
cv.destroyAllWindows()
