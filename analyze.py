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

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Start processing frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
        edges = cv2.Canny(gray, thr * 1.0, thr * 0.5)
        cv2.imshow("Frame", edges)

        # End processing frame
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
