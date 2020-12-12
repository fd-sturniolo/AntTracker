from classes import *
import cv2 as cv
import numpy as np
import argparse
import datetime
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument('filename',type=str,help="[Nombre de archivo]{.mp4,.tag,.rtg}")
parser.add_argument('--draw','-d',action="store_const",const=True,default=False,help="Dibujar trayectorias")
parser.add_argument('--save','-s',action="store_const",const=True,default=False,help="Guardar trayectorias como imágenes")

args = parser.parse_args()

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
def drawTrajectory(trajectory: List[Rect], img, antId):
    color = getNextColor.forId(antId)
    points = [rect.center() for rect in trajectory]

    points = np.int0(points)
    return cv.polylines(img, [points], False, color, 1)
def toTuple(point: Vector) -> Tuple[int,int]:
    return tuple(point.astype(int))

filename = args.filename
draw = args.draw
saveFrames = args.save

statsDict = dict({
    "filename": filename,
    "ants": [],
    "goingUpAvgVel": 0,
    "goingDnAvgVel": 0,
    "goingUpN": 0,
    "goingDnN": 0
    })

tracker = Tracker.deserialize(filename=f"./{filename}-tracked.rtg")
# tracker = Tracker.deserialize(filename=f"./{filename}-labeled.rtg")

if draw or saveFrames:
    video = cv.VideoCapture(f"./{filename}.mp4")
    _,originalFrame = video.read()
if saveFrames:
    framesToSave = []

crossingAnts = chain(tracker.getAntsThatCrossed(CrossDirection.GoingDown),tracker.getAntsThatCrossed(CrossDirection.GoingUp))

goingUpAvgVel = np.ndarray(2)
goingUpN = 0
goingDnAvgVel = np.ndarray(2)
goingDnN = 0

ant: TrackedAnt
for ant in crossingAnts:
    frameAndVels, averageVel, averageSpeed = ant.getVelocity()
    maxSpeed = max([np.linalg.norm(vel) for vel in frameAndVels[1]])
    direction = tracker.getCrossDirection(ant)
    trajectory = ant.getTrajectory()
    rect: Rect
    rectSizes = [rect.size() for rect in trajectory]
    avgSize = np.mean(rectSizes)
    medianSize = np.median(rectSizes)
    stdShape = np.std([rect.ratio() for rect in trajectory])
    stdSize = np.std(rectSizes)
    leafHolding = ant.isHoldingLeaf() == HoldsLeaf.Yes

    if direction == CrossDirection.GoingUp:
        goingUpAvgVel += averageVel
        goingUpN += 1
    if direction == CrossDirection.GoingDown:
        goingDnAvgVel += averageVel
        goingDnN += 1

    print(f"{'[LEAF]' if leafHolding else '[NOLF]'} ID: {ant.id}. Avg. Vel: {averageVel}. Avg. Spd: {averageSpeed}.")
    statsDict["ants"].append(dict({
        "id":ant.id,
        "avgVel":averageVel.tolist(),
        "avgSpd":averageSpeed,
        "maxSpd":maxSpeed,
        "avgSize":avgSize,
        "medianSize":medianSize,
        "stdSize":stdSize,
        "stdShape":stdShape,
        "direction":direction,
        "leafHolding":leafHolding
        }))

    if draw or saveFrames:
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
        if draw:
            cv.imshow(str(ant.id),firstFrame)
        if saveFrames:
            framesToSave.append((ant.id,firstFrame))

if draw: k = cv.waitKey(00) & 0xff

goingUpAvgVel /= goingUpN
goingDnAvgVel /= goingDnN

statsDict["goingUpAvgVel"] = goingUpAvgVel
statsDict["goingDnAvgVel"] = goingDnAvgVel
statsDict["goingUpN"] = goingUpN
statsDict["goingDnN"] = goingDnN


statsJson = ujson.dumps(statsDict,indent=2)

tstamp = int(datetime.datetime.now(tz=None).timestamp())
folder = f"./tracked-{filename}-{tstamp}"
from os import mkdir
mkdir(folder)
with open(f"{folder}/data.log", "w") as datafile:
    datafile.write(statsJson)
if saveFrames:
    for antId,frame in framesToSave:
        cv.imwrite(f"{folder}/{antId}.jpg",frame)

