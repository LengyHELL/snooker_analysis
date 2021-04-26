#!/usr/bin/env python3

import sys
import numpy as np
import cv2
import time
import math

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

def in_color_range(color, range, offset):
    low = 1 - offset
    high = 1 + offset
    ret = (low * range) <= color <= (high * range)
    return ret

def quantize_image(img, K):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))

def get_color_class(color):
    colors = []
    colors.append([(32, 217, 220), "yellow"])
    colors.append([(27, 198, 151), "brown"])
    colors.append([(77, 193, 191), "green"])
    colors.append([(94, 205, 172), "blue"])
    colors.append([(36, 133, 220), "pink"])
    colors.append([(54, 218, 47), "black"])
    colors.append([(33, 53, 253), "white"])
    colors.append([(12, 218, 131), "red"])
    min = 255 * 3
    index = -1
    for i in range(len(colors)):
        value = abs(colors[i][0][0] - color[0])
        value += abs(colors[i][0][1] - color[1])
        value += abs(colors[i][0][2] - color[2])
        if value < min:
            min = value
            index = i

    return colors[index]

def intersection(o1, p1, o2, p2):
    x = [o2[0] - o1[0], o2[1] - o1[1]]
    d1 = [p1[0] - o1[0], p1[1] - o1[1]]
    d2 = [p2[0] - o2[0], p2[1] - o2[1]]

    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if (abs(cross) < 1e-8)
        return False

    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    r = [o1[0] + d1[0] * t1, o1[1] + d1[1] * t1]
    return r

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
                test = [h[0] for h in hull]
                test2 = [[x, x + y, y, x - y, -x, -x - y, -y, -x + y]  for x, y in test]
                points = [0, 0, 0, 0, 0, 0, 0, 0]
                maxes = test2[0]
                #(maximize x, x+y, y, x-y, -x, -x-y, -y, -x+y)
                for i in range(len(test2)):
                    for j in range(8):
                        if test2[i][j] > maxes[j]:
                            points[j] = i
                            maxes[j] = test2[i][j]

                oct = []
                for i in range(len(test)):
                    if i in points:
                        oct.append(test[i])

                oct = np.array(oct)
                print(oct)

                table_area = cv2.contourArea(hull)

                neg = cv2.bitwise_not(edges)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                neg = cv2.erode(neg, kernel, iterations = 1)
                cnt, hier = cv2.findContours(neg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                # (table area / 775) <= ball area <= (table area / 760)
                min_area = table_area / 7000 #1200
                max_area = table_area / 1000 #700
                balls = []
                colors = []
                positions = []

                for c in cnt:
                    c = cv2.convexHull(c)
                    (x, y), r = cv2.minEnclosingCircle(c)
                    circle = r * r * math.pi
                    area = cv2.contourArea(c)
                    rate = 0.5
                    if ((1 - rate) * circle) <= area <= ((1 + rate) * circle):
                        if min_area <= cv2.contourArea(c) <= max_area:
                            balls.append(c)
                            positions.append((x, y))

                            ball_mask = np.zeros(edges.shape, np.uint8)
                            cv2.drawContours(ball_mask, [c], -1, 255, -1)
                            cv2.drawContours(ball_mask, [c], -1, 0, 1)
                            color = cv2.mean(hsv, mask = ball_mask)
                            colors.append(color)

                cv2.drawContours(frame, [oct], -1, (255, 0, 0), 2)

                for i in range(len(balls)):
                    c = get_color_class(colors[i])[1]
                    color = (0, 0, 0)
                    if c == "yellow":
                        color = (0, 255, 255)
                    elif c == "brown":
                        color = (0, 127, 127)
                    elif c == "green":
                        color = (0, 255, 0)
                    elif c == "blue":
                        color = (255, 0, 0)
                    elif c == "pink":
                        color = (255, 0, 255)
                    elif c == "black":
                        color = (0, 0, 0)
                    elif c == "white":
                        color = (255, 255, 255)
                    elif c == "red":
                        color = (0, 0, 255)

                    cv2.drawContours(frame, [balls[i]], -1, color, 2)

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
