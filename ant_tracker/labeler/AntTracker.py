from classes import *
from PreLabeler import labelVideo
from os.path import splitext, exists # for file management
from typing import Dict
from enum import Enum, auto
import itertools
import cv2 as cv

class Method(Enum):
    MEANSHIFT=auto(),
    CAMSHIFT=auto(),
    KALMAN=auto(),
    PYRLK=auto(),

filename = "dia"
reset_tags = False
method = Method.KALMAN
showInProgressTracking = True
lastFrameToTrack = 200 ## `None` o un número
old = True

file = f"{filename}.mp4"
video = cv.VideoCapture(file)
vshape = (int(video.get(cv.CAP_PROP_FRAME_HEIGHT)),int(video.get(cv.CAP_PROP_FRAME_WIDTH)))
vlen = int(video.get(cv.CAP_PROP_FRAME_COUNT))

# Nombre default de archivo de etiquetas: nombre_video.tag
tagfile = f"{filename}-untagged.tag"

def new_tags(file: str, tagfile: str) -> AntCollection:
    antCollection = AntCollection(np.empty(vshape))
    for frame,mask in enumerate(labelVideo(file)):
        antCollection.addUnlabeledFrame(frame,mask)
    antCollection.videoSize = mask.size
    antCollection.videoShape = tuple(mask.shape)
    jsonstring = antCollection.serialize()

    with open(tagfile,"w") as file:
        file.write(jsonstring)
    return antCollection

if reset_tags:
    antCollection = new_tags(file,tagfile)
else:
    try:
        antCollection = AntCollection.deserialize(filename=tagfile)
    except:
        antCollection = new_tags(file,tagfile)

# cv.namedWindow("regions",cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)
# cv.namedWindow("mask",cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)
if showInProgressTracking:
    cv.namedWindow("img",cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)
# cv.namedWindow("original",cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

box_stays_threshold = 0.05
height_ratio = 0.1
width_ratio = 0.1
            # left, top, width, height
image_rect  = Rect(0, 0, vshape[1], vshape[0])
zone_up     = Rect(
    0,
    0,
    vshape[1],
    int(vshape[0]*height_ratio)
)
zone_down   = Rect(
    0,
    int(vshape[0]*(1-height_ratio)),
    vshape[1],
    int(vshape[0]*height_ratio)
)
zone_left   = Rect(
    0,
    int(vshape[0]*height_ratio),
    int(vshape[1]*width_ratio),
    int(vshape[0]*(1-2*height_ratio))
)
zone_right  = Rect(
    int(vshape[1]*(1-width_ratio)),
    int(vshape[0]*height_ratio),
    int(vshape[1]*width_ratio),
    int(vshape[0]*(1-2*height_ratio))
)
zone_middle = Rect(
    int(vshape[1]*width_ratio),
    int(vshape[0]*height_ratio),
    int(vshape[1]*(1-2*width_ratio)),
    int(vshape[0]*(1-2*height_ratio))
)

idList = []
def first_available(l: List[int]) -> int:
    c = 1
    while c in l:
        c += 1
    return c
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

def get_new_boxes(unlabeleds: ColoredMaskWithUnlabel, exclude: Rect, frame: int) -> List[Dict]:
    # remove ants in exclude
    x,y,w,h = exclude.unpack()
    zone_regions = unlabeleds.copy()
    # zone_regions = cv.rectangle(zone_regions, (x,y), (x+w,y+h), 0,-1)
    # cv.imshow("regions",zone_regions*255)

    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(zone_regions)
    nlabels, stats, centroids = nlabels-1, stats[1:], centroids[1:]

    boxes = []

    for label in range(nlabels):
        # rect = Rect(
        #     stats[label][cv.CC_STAT_LEFT]-3 if stats[label][cv.CC_STAT_LEFT]-3>=0 else 0,
        #     stats[label][cv.CC_STAT_TOP]-3 if stats[label][cv.CC_STAT_TOP]-3>=0 else 0,
        #     stats[label][cv.CC_STAT_WIDTH]+6,
        #     stats[label][cv.CC_STAT_HEIGHT]+6)
        rect = Rect(
            stats[label][cv.CC_STAT_LEFT],
            stats[label][cv.CC_STAT_TOP],
            stats[label][cv.CC_STAT_WIDTH],
            stats[label][cv.CC_STAT_HEIGHT],
            frame)
        # if rect.is_in_boundary_of(image_rect): continue

        x,y,w,h = rect.unpack()
        roi = originalFrame[y:y+h, x:x+w]
        hsv_roi = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

        mask = labels[y:y+h, x:x+w]
        mask[mask==0] = 0
        mask[mask!=0] = 255
        mask = mask.astype('uint8')

        hist = cv.calcHist([hsv_roi],[2],mask,[255],[0,255])
        cv.normalize(hist,hist,0,255,cv.NORM_MINMAX)
        antId = 1
        # if idList == []:
        #     antId = 1
        # else:
        #     antId = first_available(idList)
        boxes.append(dict(id=antId,rect=rect,hist=hist))
        idList.append(antId)
    return boxes

def draw_boxes(frame,name,boxes):
    img = frame.copy()
    for box in boxes:
        x,y,w,h = box["rect"].unpack()
        label = box["id"]
        img = cv.rectangle(img, (x,y), (x+w,y+h), getNextColor.kelly_colors[label%len(getNextColor.kelly_colors)],2)
        img = cv.putText(img,str(label),(x,y),cv.FONT_HERSHEY_SIMPLEX,1,255)
    # cv.imshow(name,img)
    pass

_,originalFrame = video.read()
firstFrame = originalFrame.copy()
_,unlabeledFrame = antCollection.getUnlabeled(0)
lastFrame = antCollection.getLastFrame() if lastFrameToTrack is None else lastFrameToTrack
unlabeleds = antCollection.getUnlabeledMask(unlabeledFrame)
boxes = get_new_boxes(unlabeleds,zone_middle,0)

tracker = Tracker(vlen,vshape,minDistanceBetween=10)

tracker.add_new_ants([box["rect"] for box in boxes],zone_middle,image_rect)


if method == Method.KALMAN:
    kalman = cv.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1,0,0,0],
                                        [0,1,0,0]],np.float32)

    kalman.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]],np.float32)

    kalman.processNoiseCov = np.array([[1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32) * 0.03

    measurement = np.array((2,1), np.float32)
    prediction = np.zeros((2,1), np.float32)

def draw_ants(img, ants: List[TrackedAnt], frame: int):
    for ant in ants:
        color = getNextColor.kelly_colors[ant.id%len(getNextColor.kelly_colors)]
        rect: Rect = first(ant.rects, lambda r: r.frame == frame)
        if rect is not None:
            x,y,w,h = rect.unpack()
            img = cv.rectangle(img, (x,y), (x+w,y+h), color, 2 if rect.overlaps(zone_middle) else 1)
            img = cv.putText(img,str(ant.id),(x,y),cv.FONT_HERSHEY_SIMPLEX,1,255)
            img = cv.putText(img,TrackingState.toString(ant.state,True),(x-10,y-3),cv.FONT_HERSHEY_SIMPLEX,1,255)
    return img

for frame in range(1,lastFrame):
    _,originalFrame = video.read()
    _,unlabeledFrame = antCollection.getUnlabeled(frame)
    unlabeleds = antCollection.getUnlabeledMask(unlabeledFrame)
    new_possib_boxes = get_new_boxes(unlabeleds, zone_middle, frame)

    if not old:
        draw_boxes(originalFrame,'boxes',new_possib_boxes)
        tracker.add_new_ants([box["rect"] for box in new_possib_boxes],zone_middle,image_rect)
        # print(tracker)

        img = originalFrame.copy()
        x,y,w,h = zone_up.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 0,2)
        x,y,w,h = zone_down.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 0,2)
        x,y,w,h = zone_left.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 0,2)
        x,y,w,h = zone_right.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 0,2)
        x,y,w,h = zone_middle.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 127,2)
        img = draw_ants(img,tracker.trackedAnts,frame)

        if showInProgressTracking:
            cv.imshow('trackedAnts',img)

    stillboxes = []
    # print(idList)


    if old:

        ## Se descartan las cajas que tengan color de fondo
        for box in boxes:
            x,y,w,h = box["rect"].unpack()
            roi = unlabeleds[y:y+h, x:x+w]

            if np.mean(roi) > box_stays_threshold:
            # if np.median(roi) > 0.5:
                stillboxes.append(box)
            else:
                idList.remove(box["id"])
        boxes = stillboxes

        # draw_boxes(originalFrame,'new possib boxes',new_possib_boxes)
        ## Agregamos las cajas nuevas (que no se superpongan con las viejas)
        new_boxes = boxes.copy()
        for newbox in new_possib_boxes:
            # if is_on_zone_boundary(bx,by,bw,bh): continue
            if not any((box["rect"].overlaps(newbox["rect"]) for box in boxes)):
                new_boxes.append(newbox)
            else:
                idList.remove(newbox["id"])
            # overlap = False
            # for box in boxes:
            #     overlap = box["rect"].overlaps(newbox["rect"])
            #     if overlap:
            #         print("overlap: ", box["id"], newbox["id"])
            #         break
            # if not overlap: new_boxes.append(newbox)
        boxes = new_boxes

        # draw_boxes(originalFrame,'boxes before update',boxes)


        img = originalFrame.copy()
        x,y,w,h = zone_up.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 0,2)
        x,y,w,h = zone_down.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 0,2)
        x,y,w,h = zone_left.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 0,2)
        x,y,w,h = zone_right.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 0,2)
        x,y,w,h = zone_middle.unpack()
        img = cv.rectangle(img, (x,y), (x+w,y+h), 127,2)

        # _,unlabeledFrame = antCollection.getUnlabeled(frame)
        # mask = antCollection.getUnlabeledMask(unlabeledFrame)
        for box in boxes:
            hist,label = box["hist"],box["id"]
            color = getNextColor.kelly_colors[label%len(getNextColor.kelly_colors)]
            hsv = cv.cvtColor(originalFrame,cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv],[2],hist,[0,255],1)

            # print("before: ", bounding_rect)
            rect: Rect
            rect = box["rect"]
            boxtuple = rect.unpack()

            if method == Method.CAMSHIFT:
                track_box, boxtuple = cv.CamShift(dst, boxtuple, term_crit)
                pts = cv.boxPoints(track_box)
                pts = np.int0(pts)
                img = cv.polylines(img,[pts], True, color, 2)
                x,y,w,h = boxtuple
                box["rect"] = Rect(x,y,w,h)
            elif method == Method.MEANSHIFT:
                track_box, boxtuple = cv.meanShift(dst, boxtuple, term_crit)
                x,y,w,h = boxtuple
                box["rect"] = Rect(x,y,w,h)
                img = cv.rectangle(img, (x,y), (x+w,y+h), color, 2 if box["rect"].overlaps(zone_middle) else 1)
            elif method == Method.KALMAN:
                center = rect.center().astype(np.float32)
                kalman.correct(center)
                prediction = kalman.predict()
                x = np.int(prediction[0]-(0.5*w))
                y = np.int(prediction[1]-(0.5*h))
                
                # w = np.int(prediction[0]+(0.5*w))
                # h = np.int(prediction[1]+(0.5*h))
                box["rect"] = Rect(x,y,w,h)
                img = cv.rectangle(img, (x,y), (x+w,y+h), color,2)

            img = cv.putText(img,str(label),(x,y),cv.FONT_HERSHEY_SIMPLEX,1,255)
        # cv.imshow('mask',unlabeleds*255)
        # cv.imshow('original',originalFrame)
    if (showInProgressTracking):
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xff
        if k == 27:
            break

# with open(f"./{filename}-tracked.rtg","w") as target:
#     target.write(tracker.serialize())

def toTuple(point: Vector) -> Tuple[int,int]:
    return tuple(point.astype(int))
def drawTrajectory(trajectory: List[Rect], img, antId):
    color = getNextColor.forId(antId)
    points = [rect.center() for rect in trajectory]

    points = np.int0(points)
    return cv.polylines(img, [points], False, color, 1)
from itertools import chain

## Hay tres casos:
##      una hormiga se queda quieta en el medio (state = Ant)
##      una hormiga va al medio y vuelve (state = Left)
##      una hormiga queda muchos frames sin reconocer en el medio (state = Ant) y después se reconoce (state = any)
if not old:
    ants = list(tracker.getAntsThatDidntCross())
    print(len(ants))

    video = cv.VideoCapture(f"./{filename}.mp4")
    _,originalFrame = video.read()

    for ant in ants:
        frameAndVels, averageVel, averageSpeed = ant.getVelocity()
        if frameAndVels == []: continue
        maxSpeed = max([np.linalg.norm(vel) for frame,vel in frameAndVels])
        direction = tracker.getCrossDirection(ant)
        trajectory = ant.getTrajectory()
        rect: Rect
        rectSizes = [rect.size() for rect in trajectory]
        avgSize = np.mean(rectSizes)
        medianSize = np.median(rectSizes)
        stdShape = np.std([rect.ratio() for rect in trajectory])
        stdSize = np.std(rectSizes)
        leafHolding = ant.isHoldingLeaf() == HoldsLeaf.Yes

        firstFrame = originalFrame.copy()
        firstFrame = cv.putText(firstFrame,'[LEAF]' if leafHolding else '[NOLF]',(20,20),cv.FONT_HERSHEY_SIMPLEX,0.3,255)
        firstFrame = cv.putText(firstFrame,f"Av.Vel: {averageVel}",(20,40),cv.FONT_HERSHEY_SIMPLEX,0.3,255)
        firstFrame = cv.putText(firstFrame,f"Av.Spd: {averageSpeed}",(20,60),cv.FONT_HERSHEY_SIMPLEX,0.3,255)
        cv.arrowedLine(firstFrame, (20,80), toTuple((20,80) + averageVel*5), (0,0,0), 1, tipLength=.3)

        for frame,vel in frameAndVels:
            rect1 = ant.getRectAtFrame(frame)
            pt1 = toTuple(rect1.center())
            pt2 = toTuple(rect1.center() + vel)
            cv.arrowedLine(firstFrame, pt1, pt2, (0,0,0), 1, tipLength=1)
        firstFrame = drawTrajectory(ant.getTrajectory(),firstFrame,ant.id)
        # firstFrame = cv.putText(firstFrame,str(ant.id),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,255)
        cv.imshow(str(ant.id),firstFrame)
    k = cv.waitKey(0) & 0xff

    # trackerJson = AntCollection.deserialize(filename="dia.tag").serializeAsTracker()
    # with open("./dia-labeled.rtg","w") as target:
    #     target.write(trackerJson)
