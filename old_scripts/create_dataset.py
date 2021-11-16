import os

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
for f in files:
    img = load_image(dir + "/" + f)
    cnt = find_table(img)
    img = cut_and_warp(img, cnt, (1024, 512))
    img_orig = img.copy()

    tmp = load_image("misc/templates/black_ball_hd.png")
    black = search_template(img, tmp, threshold=0.855, min_diff=15) #ok

    tmp = load_image("misc/templates/blue_ball_hd.png")
    blue = search_template(img, tmp, threshold=0.92, min_diff=15) #ok

    tmp = load_image("misc/templates/brown_ball_hd.png")
    brown = search_template(img, tmp, threshold=0.905, min_diff=15) #ok

    tmp = load_image("misc/templates/green_ball_hd.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
    green = search_template(img, tmp, threshold=0.985, min_diff=15) #ok
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    tmp = load_image("misc/templates/pink_ball_hd.png")
    pink = search_template(img, tmp, threshold=0.97, min_diff=15) #ok

    tmp = load_image("misc/templates/red_ball_hd.png")
    red = search_template(img, tmp, threshold=0.92, min_diff=15) #ok

    tmp = load_image("misc/templates/white_ball_hd.png")
    white = search_template(img, tmp, threshold=0.97, min_diff=15) #ok

    tmp = load_image("misc/templates/yellow_ball_hd.png")
    yellow = search_template(img, tmp, threshold=0.95, min_diff=15) #ok

    # correction for pesky brown ball
    temp = brown
    brown = []
    for t in temp:
        add = True
        for r in red:
            diff = calculate_diff(t, r)
            if diff < 15:
                add = False
        if add:
            brown.append(t)
    brown = np.array(brown)
    points = brown
    for p in points:
        p = (p[0] - 10, p[1] - 10)
        cv2.rectangle(img, p, (p[0] + 20, p[1] + 20), (0, 0, 255), 2)
    brown = np.array(brown)

    balls = []
    if (len(black) > 0):
        balls.append(black[0])
    else:
        balls.append([0, 0])

    if (len(blue) > 0):
        balls.append(blue[0])
    else:
        balls.append([0, 0])

    if (len(brown) > 0):
        balls.append(brown[0])
    else:
        balls.append([0, 0])

    if (len(green) > 0):
        balls.append(green[0])
    else:
        balls.append([0, 0])

    if (len(pink) > 0):
        balls.append(pink[0])
    else:
        balls.append([0, 0])

    if (len(white) > 0):
        balls.append(white[0])
    else:
        balls.append([0, 0])

    if (len(yellow) > 0):
        balls.append(yellow[0])
    else:
        balls.append([0, 0])

    for i in range(15):
        if i < len(red):
            balls.append(red[i])
        else:
            balls.append([0, 0])
    balls = np.array(balls)

    name = f.split(".")[0]
    cv2.imwrite("./misc/dataset/images/" + name + ".png", img_orig)
    np.save("./misc/dataset/labels/" + name + ".npy", balls)
    file = open("./misc/dataset/info.txt", "a")
    file.write(name + "\n")
    file.close()
