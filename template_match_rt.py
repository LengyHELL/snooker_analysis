import os
import sys

import time
import numpy as np

from mss import mss

import cv2
from analyze import label_cuts_tm, load_image, find_table, cut_and_warp, find_circles, cut_circles

index = 0
circle_radius = 9
delta = 10
timer = 0
frame_time = 0
start_time = 0

labels = ["black", "blue", "brown", "green", "pink", "red", "white", "yellow"]
templates = []
templates.append(load_image("misc/templates/black_ball_hd.png"))
templates.append(load_image("misc/templates/blue_ball_hd.png"))
templates.append(load_image("misc/templates/brown_ball_hd.png"))
templates.append(load_image("misc/templates/green_ball_hd.png"))
templates.append(load_image("misc/templates/pink_ball_hd.png"))
templates.append(load_image("misc/templates/red_ball_hd.png"))
templates.append(load_image("misc/templates/white_ball_hd.png"))
templates.append(load_image("misc/templates/yellow_ball_hd.png"))

bounding_box = {'top': 100, 'left': 100, 'width': 900, 'height': 450}
sct = mss()

start = True

while True:
    frame_time = time.time() - start_time
    start_time = time.time()
    if start:
        start = False
        timer = 0
    else:
        timer = timer + frame_time

    sct_img = sct.grab(bounding_box)
    img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

    try:
        cnt = find_table(img)
        if cnt is None:
            raise Exception("Table not found!")

        img = cut_and_warp(img, cnt, (1024, 512))
        out = img.copy()
        circles = find_circles(img, circle_radius)
        if circles is None:
            raise Exception("Failed to find circles!")

        cuts, rects = cut_circles(img, circles, circle_radius, mode="bgr")

        ids, _ = label_cuts_tm(cuts, templates)
        if ids is None:
            raise Exception("Failed to label candidates!")

        for i, id in enumerate(ids):
            color = labels[id]
            x, y = rects[i][0]
            cv2.rectangle(out, rects[i][0], rects[i][1], (0, 0, 255), 2)
            cv2.putText(out, color, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if (timer >= delta):
                cv2.imwrite("./misc/dataset4/" + color + "/" + str(index) + ".png", cuts[i])
                index += 1
        if (timer >= delta):
            timer = timer - delta
            print("Collected samples:", index)

    except Exception as e:
        cv2.imshow("screen", cv2.resize(img, (900, 450)))
    else:
        cv2.imshow("screen", cv2.resize(out, (900, 450)))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
print("Total collected:", index)