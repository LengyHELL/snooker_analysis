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

labels = ["black", "blue", "brown", "green", "pink", "red", "white", "yellow"]
templates = []
templates.append([0, load_image("misc/templates/black_ball_hd.png"), 0.855, False])
templates.append([1, load_image("misc/templates/blue_ball_hd.png"), 0.92, False])
templates.append([2, load_image("misc/templates/brown_ball_hd.png"), 0.905, False])
templates.append([3, cv2.cvtColor(load_image("misc/templates/green_ball_hd.png"), cv2.COLOR_BGR2HSV), 0.985, True])
templates.append([4, load_image("misc/templates/pink_ball_hd.png"), 0.97, False])
templates.append([5, load_image("misc/templates/red_ball_hd.png"), 0.92, False])
templates.append([6, load_image("misc/templates/white_ball_hd.png"), 0.97, False])
templates.append([7, load_image("misc/templates/yellow_ball_hd.png"), 0.95, False])

for f in files:
    start_time = time.time()
    part_time = time.time()

    img = load_image(dir + "/" + f)
    cnt = find_table(img)
    img = cut_and_warp(img, cnt, (1024, 512))
    img_orig = img.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    print("%-30s | %d ms" % ("Image processed", (time.time() - part_time) * 1000))
    part_time = time.time()

    points = []
    for t in templates:
        if t[3]:
            pts = search_template(img_hsv, t[1], threshold=t[2], min_diff=8)
        else:
            pts = search_template(img, t[1], threshold=t[2], min_diff=8)

        ids = np.uint0(np.ones((pts.shape[0], 1)) * t[0])
        pts = np.c_[pts, ids].astype("uint16")
        points.append(pts)

    # correction for brown ball
    i = 0
    while i < len(points[2]):
        remove = False
        for r in points[5]:
            diff = calculate_diff(points[2][i], r)
            if diff < 15:
                remove = True
        if remove:
            points[2] = np.delete(points[2], i, 0)
        else:
            i += 1

    print("%-30s | %d ms" % ("Templates found", (time.time() - part_time) * 1000))
    part_time = time.time()

    temp = points
    points = []
    for t in temp:
        points.extend(t)
    points = np.array(points)


    for p in points:
        p = (p[0] - 10, p[1] - 10)
        cv2.rectangle(img, p, (p[0] + 20, p[1] + 20), (0, 0, 255), 2)

    print("%-30s | %d ms" % ("Rectangles drawn", (time.time() - part_time) * 1000))
    print("%-27s -> | %d ms" % ("Total", (time.time() - start_time) * 1000))
    print("(%3d/%3d) %10s - %3d" % (index, len(files), f, len(points)))

    #file = open("./misc/out.txt", "a")
    #file.write("(%3d/%3d) %10s - %3d\n" % (index, len(files), f, len(points)))
    #for p in points:
    #    #file.write("%4d - %4d  %s\n" % (p[0], p[1], labels[p[2]]))
    #    file.write("%s\n" % (labels[p[2]]))
    #file.close()
    index += 1

    #plt.figure(figsize=(16, 6))

    #plt.subplot(121)
    #plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    #plt.xticks([])
    #plt.yticks([])

    #plt.subplot(122)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
