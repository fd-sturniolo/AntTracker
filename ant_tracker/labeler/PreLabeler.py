import sys  # for progress bar
import cv2 as cv
import numpy as np

from .classes import *

def fill_holes(img, kernel):
    Im = img // 255
    Ic = 1 - Im

    F = np.zeros_like(Im)
    F[:, 0] = Ic[:, 0]
    F[:, -1] = Ic[:, -1]
    F[0, :] = Ic[0, :]
    F[-1, :] = Ic[-1, :]

    dif = np.zeros_like(img).astype(bool)
    while np.any(~dif):
        Fnew = cv.dilate(F, kernel) * Ic
        dif = F == Fnew
        F = Fnew
    return (1 - F) * 255

def minimum_ant_area(min_radius):
    return np.pi * min_radius ** 2

# noinspection PyShadowingNames,DuplicatedCode
def labelVideo(file, metodo="mog2", roi=None, historia=50, ant_thresh=50, sheet_thresh=150, discard_percentage=.8,
               minimum_ant_radius=4, start_frame=0) -> np.ndarray:
    lastmask = None
    video = cv.VideoCapture(file)
    length = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    i = 0

    toolbar_width = 40
    # setup toolbar
    progressMsg = "Procesando 0/%d" % length
    sys.stdout.write(progressMsg + "[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (len(progressMsg) + toolbar_width + 2))  # return to start of line, after '['

    if metodo == "mediana":
        last_frames = []
        while video.isOpened():
            ret, frame = video.read()
            if ret == 0:
                break
            if roi is not None:
                frame = frame[roi]
            if i < historia:
                last_frames.append(frame)
            else:
                last_frames[:-1] = last_frames[1:]
                last_frames[-1] = frame
            i += 1

            if len(last_frames) != 1:
                fondo = np.median(last_frames, axis=0).astype('uint8')
            else:
                fondo = frame

            frame_sin_fondo = cv.absdiff(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.cvtColor(fondo, cv.COLOR_BGR2GRAY))

            _, mask = cv.threshold(frame_sin_fondo, ant_thresh, 255, cv.THRESH_BINARY)

            # Eliminar la parte de la máscara correspondiente a la lámina
            objects = frame.copy()
            objects[mask == 0, ...] = (0, 0, 0)
            objects = cv.cvtColor(objects, cv.COLOR_BGR2GRAY)
            _, lamina = cv.threshold(objects, sheet_thresh, 255, cv.THRESH_BINARY)
            # thresh = np.mean(cv.cvtColor(frame,cv.COLOR_BGR2GRAY))*background_ratio
            # _,lamina = cv.threshold(objects,thresh,255,cv.THRESH_BINARY)
            mask = cv.subtract(mask, lamina)

            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))).astype(
                'int16')

            # Descartar la máscara si está llena de movimiento (se movió la cámara!)
            if np.count_nonzero(mask) > np.size(mask) * discard_percentage:
                yield np.zeros(mask.shape, dtype='int') if lastmask is None else lastmask
                continue

            mask[mask != 0] = 1

            progress = toolbar_width * i // length + 1
            progressMsg = "Procesando %d/%d " % (i + 1, length)
            sys.stdout.write(progressMsg)
            sys.stdout.write("[%s%s]" % ("-" * progress, " " * (toolbar_width - progress)))
            sys.stdout.flush()
            sys.stdout.write("\b" * (len(progressMsg) + toolbar_width + 2))  # return to start of line, after '['

            lastmask = cv.morphologyEx(mask.copy(), cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)))
            yield mask
    elif metodo in ["mog2", "gsoc"]:
        if metodo == "mog2":
            subt = cv.createBackgroundSubtractorMOG2(detectShadows=False, history=historia)
        if metodo == "gsoc":
            subt = cv.bgsegm.createBackgroundSubtractorGSOC(replaceRate=0.0002 * historia)
        while video.isOpened():
            ret, frame = video.read()
            if ret == 0:
                break

            i += 1
            if i - 1 < max(start_frame - 50, 0):
                continue

            if roi is not None:
                frame = frame[roi]

            # noinspection PyUnboundLocalVariable
            mask = subt.apply(frame, 0)

            if i - 1 < start_frame:
                continue

            # Eliminar la parte de la máscara correspondiente a la lámina
            objects = frame.copy()
            objects[mask == 0, ...] = (0, 0, 0)
            objects = cv.cvtColor(objects, cv.COLOR_BGR2GRAY)
            _, lamina = cv.threshold(objects, sheet_thresh, 255, cv.THRESH_BINARY)

            mask = cv.subtract(mask, lamina)

            r = minimum_ant_radius
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (int(r), int(r))))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN,
                                   cv.getStructuringElement(cv.MORPH_ELLIPSE, (int(r * .8), int(r * .8))))
            # mask = cv.morphologyEx(mask,cv.MORPH_DILATE,
            #   cv.getStructuringElement(cv.MORPH_ELLIPSE,(int(r*.8),int(r*.8))))
            # Rellenar huecos para mejor detección
            mask = fill_holes(mask, cv.getStructuringElement(cv.MORPH_CROSS, (int(r), int(r))))

            # Descartar la máscara si está llena de movimiento (se movió la cámara!)
            if np.count_nonzero(mask) > np.size(mask) * discard_percentage:
                yield np.zeros(mask.shape, dtype='int') if lastmask is None else lastmask
                continue

            contours, _ = cv.findContours(mask.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)

            mask = np.zeros_like(mask)
            min_area = minimum_ant_area(minimum_ant_radius)
            contours = [cv.approxPolyDP(cont, 0.01, True) for cont in contours if cv.contourArea(cont) > min_area]
            mask = cv.fillPoly(mask, contours, 1)

            progress = toolbar_width * i // length + 1
            progressMsg = "Procesando %d/%d " % (i + 1, length)
            sys.stdout.write(progressMsg)
            sys.stdout.write("[%s%s]" % ("-" * progress, " " * (toolbar_width - progress)))
            sys.stdout.flush()
            sys.stdout.write("\b" * (len(progressMsg) + toolbar_width + 2))  # return to start of line, after '['

            lastmask = cv.morphologyEx(mask.copy(), cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)))
            # print(f"frame {i-1}")
            yield mask
    # elif metodo == "log":
    #     def toTuple(point: Vector) -> Tuple[int,int]:
    #         return tuple(point.astype(int))
    #     while(video.isOpened()):
    #         ret, frame = video.read()
    #         if ret == 0:
    #             break
    #         if roi != None:
    #             frame = frame[roi]
    #         i += 1
    #
    #         frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #
    #         gaussian = cv.GaussianBlur(frame,(7,7),500)
    #
    #         log = cv.Laplacian(gaussian,cv.CV_32F,None,7)
    #         log_norm = cv.normalize(log,None,0,255,cv.NORM_MINMAX, cv.CV_8U)
    #
    #         # _,thresholded = cv.threshold(log_norm, 160, 255, cv.THRESH_TOZERO)
    #         thresholded = log.copy()
    #         thresholded[log<i*.1] = 0
    #         thresholded_norm = cv.normalize(thresholded,None,0,255,cv.NORM_MINMAX, cv.CV_8U)
    #
    #         maxim = local_minima((255-thresholded_norm),1)
    #         cv.imshow('threshinv', (255-thresholded_norm))
    #         print(len(maxim))
    #         # print(maxim)
    #
    #         # if cv.waitKey(0) & 0xff == 27:
    #         #     raise RuntimeError
    #         # cv.imshow('maxim', np.array((maxim*255)).astype('uint8'))
    #         # maxima_pos = np.argwhere(maxim)
    #
    #         located_maxima = cv.cvtColor(thresholded_norm.copy(),cv.COLOR_GRAY2BGR)
    #         for m in maxim:
    #             # print(m)
    #             located_maxima = cv.circle(located_maxima,m,4,(255,0,0),-1)
    #
    #         mask = thresholded.copy().astype('uint8')
    #         mask[mask!=0] = 1
    #
    #         # cv.imshow('frame',frame)
    #         # cv.imshow('gaussian',gaussian)
    #         # # cv.imshow('log',log)
    #         # cv.imshow('log_norm',log_norm)
    #         # cv.imshow('thresholded_norm',thresholded_norm)
    #         cv.imshow('located_maxima',located_maxima)
    #
    #         if cv.waitKey(0) & 0xff == 27:
    #             raise RuntimeError
    #
    #         progress = toolbar_width*i//length+1
    #         progressMsg = "Procesando %d/%d " % (i+1,length)
    #         sys.stdout.write(progressMsg)
    #         sys.stdout.write("[%s%s]" % ("-" * progress," " * (toolbar_width-progress)))
    #         sys.stdout.flush()
    #         sys.stdout.write("\b" * (len(progressMsg)+toolbar_width+2)) # return to start of line, after '['
    #
    #         yield mask

    sys.stdout.write("\n")  # this ends the progress bar
    video.release()
    return

if __name__ == '__main__':
    from pathlib import Path
    import os

    def valid_roi(string):
        lst = [int(i) for i in string.split(",")]
        if len(lst) != 4 or any(i < 0 for i in lst) or lst[2] == 0 or lst[3] == 0:
            raise argparse.ArgumentTypeError("%r no es un rectángulo (x,y,w,h)" % string)
        return slice(lst[1], lst[1] + lst[3]), slice(lst[0], lst[0] + lst[2])
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    import argparse

    parser = argparse.ArgumentParser(description="Segmentar hormigas con un algoritmo menos que óptimo")
    parser.add_argument('filename', help="Path al video")
    parser.add_argument('--output', '-o', type=str, default=None, metavar="O", help="Nombre del archivo de salida")
    parser.add_argument('--method', '-m', type=str, default='mog2', metavar="M",
                        help="Método de segmentación [mediana,mog2,gsoc,log]. Def: mog2")
    parser.add_argument('--hist', '-g', type=int, default=50, metavar="H", help="Número de frames de historia. Def: 50")
    parser.add_argument('--antThresh', '-a', type=int, default=50, metavar="A",
                        help="Umbral para segmentar hormigas (sólo mediana y log). Def: 50")
    parser.add_argument('--sheetThresh', '-s', type=int, default=150, metavar="S",
                        help="Umbral para remover lámina de fondo (mientras mayor el número, más sensible). Def: 150")
    parser.add_argument('--roi', '-r', type=valid_roi, default=None, metavar="rect",
                        help="Zona de interés, rectángulo \"x,y,w,h\". (default: todo el video)")
    parser.add_argument('--discard', '-d', type=float, default=0.8, metavar="D",
                        help="Porcentaje de la imagen por sobre la cual si se encuentra movimiento, "
                             "es descartada la máscara (default: 0.8)")

    args = parser.parse_args()

    if args.output is None:
        outfile = args.filename[:-3] + "tag"  # hacky
    else:
        outfile = args.output
    print("outfile: ", outfile)
    # ensure_dir("./tmp/")
    # import shutil
    # shutil.rmtree("./tmp/")
    # ensure_dir("./tmp/")
    antCollection = AntCollection(info=LabelingInfo(video_path=Path(args.filename), ants=[], unlabeled_frames=[]))
    # antCollection_dict = {"ants": [], "unlabeledFrames": [], "videoSize": 0, "videoShape": (0,0)}
    # unlabeledFrames = antCollection_dict["unlabeledFrames"]
    print("OpenCV: ", cv.__version__)
    import time

    start_time = time.process_time()
    for frame, mask in enumerate(
            labelVideo(args.filename, metodo=args.method, roi=args.roi, historia=args.hist, ant_thresh=args.antThresh,
                       sheet_thresh=args.sheetThresh, discard_percentage=args.discard)):
        antCollection.addUnlabeledFrame(frame, mask)
    print("File: %s - Method: %s" % (args.filename, args.method))
    print("Number of frames: %d" % (frame + 1))  # noqa
    print("Time elapsed: %02f seconds" % (time.process_time() - start_time))
    # print("\n%d" % frame)
    print("mask.shape", str(mask.shape))  # noqa
    print("mask.size", str(mask.size))
    antCollection.videoSize = mask.size  # noqa
    antCollection.videoShape = mask.shape
    antCollection.videoLength = frame + 1

    jsonstring = antCollection.serialize()
    antCollection2 = AntCollection.deserialize(video_path=Path(args.filename), jsonstring=jsonstring)
    if jsonstring == antCollection2.serialize():
        print("Serialization consistent")
    else:
        raise ValueError("Serialization failed")

    with open(outfile, "w") as file:
        file.write(jsonstring)
