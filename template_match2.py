import os
import sys

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

import cv2
from analyze import load_image, find_table, cut_and_warp, search_template, calculate_diff

# http://www.flyordie.hu/snooker/

#img = load_image("./misc/images/00000002.jpg")
#cnt = find_table(img)
#img = cut_and_warp(img, cnt, (1024, 512))
#cv2.imwrite("new.png", img)

dir = "./misc/images"
files = os.listdir(dir)
index = 1
index2 = 0

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
    start_time = time.time()
    img = load_image(dir + "/" + f)
    cnt = find_table(img)
    img = cut_and_warp(img, cnt, (1024, 512))

    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 15, param2=15, minRadius=5, maxRadius=11)

    cut = img.copy()
    if circles is not None:
        circles = np.round(circles[0,:]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(out, (x, y), r, (255, 0, 0), 1)
            x = x - 9
            y = y - 9
            w = 18
            h = 18
            cut = img[y:y+h, x:x+w]
            res = []
            for t in templates:
                res.append(cv2.matchTemplate(cut, t, cv2.TM_CCORR_NORMED))
            res = np.array(res).reshape(-1)
            res = np.where(res >= 0.75, res, 0)
            if res.any() > 0:
                color = labels[np.argmax(res)]
                cv2.imwrite("./misc/dataset2/" + color + "/" + str(index2) + ".png", cut)
                index2 += 1
    else:
        print("No circles found")

    #print(f, len(circles))
    print("%-27s -> | %d ms" % ("Total", (time.time() - start_time) * 1000))
    #plt.figure(figsize=(10, 5))
    #plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()
print(index2)
