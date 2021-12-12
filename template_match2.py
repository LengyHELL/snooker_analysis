import os
import sys

import time
import numpy as np

import cv2
from analyze import label_cuts_tm, load_image, find_table, cut_and_warp, find_circles, cut_circles

dir = "./misc/images"
files = os.listdir(dir)
index = 0
circle_radius = 9

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

for f in files:
    if f.split(".")[1] != "png":
        continue
    start_time = time.time()
    img = load_image(dir + "/" + f)

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
            cv2.imwrite("./misc/dataset4/" + color + "/" + str(index) + ".png", cuts[i])
            index += 1

    except Exception as e:
        print(e.args[0])
    else:
        print("%-27s -> | %d ms" % ("Total", (time.time() - start_time) * 1000))
print("Created:", index)