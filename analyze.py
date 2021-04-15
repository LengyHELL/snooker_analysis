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
