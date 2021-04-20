#!/usr/bin/env python3

import sys
import numpy as np
import cv2
import time

def get_time(start_time):
    return int((time.time() - start_time) * 1000)

def is_inside(inside, outside, limit_val=-1):
    point_limit = limit_val * len(inside)
    if limit_val < 0:
        point_limit = 1
    in_point = 0;
    for i in inside:
        is_in = cv2.pointPolygonTest(outside, tuple(i[0]), False)
        if is_in >= 0:
            in_point += 1
            if in_point >= point_limit:
                return True
    return False

def get_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    return cv2.Canny(gray, 1.0 * thr, 0.5 * thr)

start_time = time.time()

# checking arguments
arg = {}
for a in sys.argv[1:]:
    if (a[0] == "-"):
        a = a[1:]
        a = a.split("=")
        if len(a) == 2:
            arg[a[0]] = a[1]
        elif len(a) == 1:
            arg[a[0]] = ""
        else:
            sys.exit(3)
    else:
        sys.exit(2)

if "input" not in arg:
    sys.exit(1)

cap = cv2.VideoCapture(arg["input"])
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

frame_pos = 0
if "start-frame" in arg:
    frame_start = int(arg["start-frame"]) * cap.get(cv2.CAP_PROP_FPS)
    if (frame_start >= total_frames) or (frame_start < 0):
        frame_pos = 0
    else:
        frame_pos = frame_start

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

if (cap.isOpened() == False):
    print("Error opening input video!")
    sys.exit(4)

play = True
ret = True

while (cap.isOpened()):
    if ret == True:
        if play:
            ret, frame = cap.read()
            # Start processing frame

            #frame = cv2.fastNlMeansDenoisingColored(frame, None, 15, 10, 7, 21)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_green = np.array([40, 190, 50])
            upper_green = np.array([65, 255, 225])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            res = cv2.bitwise_and(frame, frame, mask = mask)

            edges = get_edges(res)
            kernel = np.array([ [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0]], dtype=np.uint8)

            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations = 1)

            kernel = np.array([ [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]], dtype=np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations = 1)

            cnt, hier = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


            if len(cnt) > 0:
                table = sorted(cnt, key=lambda x : cv2.contourArea(x), reverse=True)[0]

                hull = cv2.convexHull(table)

                # (table area / 760) <= ball area <= (table area / 775)
                table_area = cv2.contourArea(hull)
                #cnt = cnt[1:]
                #balls = []
                #for c in cnt:
                #    area = cv2.contourArea(c)
                #    lower = table_area / 1200
                #    upper = table_area / 700
                #    if lower <= area <= upper:
                #        balls.append(c)

                params = cv2.SimpleBlobDetector_Params()

                params.filterByArea = True
                params.minArea = table_area / 2000 #1200
                params.maxArea = table_area / 700 #700

                params.filterByCircularity = True
                params.minCircularity = 0.4 #0.1
                params.maxCircularity = 1

                params.filterByConvexity = True
                params.minConvexity = 0.87

                detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(edges)

                #mask = np.zeros(edges.shape, np.uint8)
                #mask = cv2.drawContours(mask, [hull], -1, 255, -1)
                #board = cv2.bitwise_and(frame, frame, mask = mask)

                cv2.drawContours(frame, [hull], -1, (125, 0, 125), 2)
                #cv2.drawContours(frame, keypoints, -1, (255, 0, 0), 1)
                frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.imshow("Frame", frame)

            # End processing frame
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        if (cv2.waitKey(33) & 0xFF == ord("a")) and (not key_lock):
            play = not play
            key_lock = True
        if not (cv2.waitKey(33) & 0xFF == ord("a")):
            key_lock = False
    else:
        break

cap.release()
cv2.destroyAllWindows()
