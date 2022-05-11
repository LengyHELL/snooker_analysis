import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys

import time
import numpy as np

import tensorflow as tf
from tensorflow import keras

import cv2
from analyze import find_circles, load_image, find_table, cut_and_warp, cut_circles, label_cuts_nn

circle_radius = 9
labels = ["black", "blue", "brown", "green", "pink", "red", "white", "yellow"]
model = keras.models.load_model("classifier_combined3.h5")

#img = load_image("misc/images/00000001.jpg")
img = load_image("misc/png/input_screen.png")
img_orig = cv2.resize(img, (800, 450))

data = []

start_time = time.time()
try:
    cnt = find_table(img)
    if cnt is None:
        raise Exception("Table not found!")
    img = cut_and_warp(img, cnt, (1024, 512))

    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = find_circles(img, circle_radius)

    if circles is None:
        raise Exception("Failed to find circles!")

    cuts, rects = cut_circles(img, circles, circle_radius, mode="combined")

    ids, _ = label_cuts_nn(cuts, model)
    if ids is None:
        raise Exception("Failed to label candidates!")

    for i, id in enumerate(ids):
        x, y = rects[i][0]
        data.append([x + circle_radius, y + circle_radius, labels[id]])
        cv2.rectangle(out, rects[i][0], rects[i][1], (0, 0, 255), 2)
        cv2.putText(out, labels[id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        #cv2.circle(out, (x + 9, y + 9), 10, (0, 0, 255), 2)
except Exception as e:
    pred = np.array([])
    cv2.imwrite("./misc/output.png", cv2.resize(img, (900, 450)))
else:
    print("No table found!")
    cv2.imwrite("./misc/output.png", out)

print("%-27s -> | %d ms" % ("Total", (time.time() - start_time) * 1000))
data.sort(key=lambda x : x[2])
for d in data:
    print("|%-10s|%-5d|%-5d|" % (d[2], d[0], d[1]))