import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys

import time
import numpy as np

import tensorflow as tf
from tensorflow import keras

import cv2
from analyze import load_image, find_table, cut_and_warp

# http://www.flyordie.hu/snooker/

labels = ["black", "blue", "brown", "green", "pink", "red", "white", "yellow"]
model = keras.models.load_model("classifier.h5")

img = load_image("misc/input_screen.png")
img_orig = cv2.resize(img, (800, 450))

data = []

start_time = time.time()
cnt = find_table(img)
if cnt is not None:
    img = cut_and_warp(img, cnt, (1024, 512))

    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 15, param1=30, param2=14, minRadius=5, maxRadius=11)

    cuts = []
    rects = []
    if circles is not None:
        circles = np.round(circles[0,:]).astype("int")

        for (x, y, r) in circles:
            x, y, w, h = (x-9, y-9, 18, 18)
            if (x >= 0) and (y >= 0) and ((x + w) < img.shape[1]) and ((y + h) < img.shape[0]):
                cuts.append(img[y:y+h, x:x+w])
                rects.append(((x, y), (x+w, y+h)))

        cuts = np.array(cuts)
        norms = np.array(cuts / 255)
        if len(norms) > 0:
            pred = model.predict(norms)
            for i in range(len(rects)):
                x, y = rects[i][0]
                xx, yy = rects[i][1]
                data.append([int((x + xx) / 2), int((y + yy) / 2), labels[np.argmax(pred[i])]])
                cv2.rectangle(out, rects[i][0], rects[i][1], (0, 0, 255), 2)
                cv2.putText(out, labels[np.argmax(pred[i])], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                #cv2.circle(out, (x + 9, y + 9), 10, (0, 0, 255), 2)
        else:
            pred = np.array([])
        cv2.imwrite("./misc/output.png", cv2.resize(out, (900, 450)))
    else:
        cv2.imwrite("./misc/output.png", cv2.resize(out, (900, 450)))
else:
    cv2.imwrite("./misc/output.png", img_orig)

print("%-27s -> | %d ms" % ("Total", (time.time() - start_time) * 1000))
data.sort(key=lambda x : x[2])
for d in data:
    print("|%-10s|%-5d|%-5d|" % (d[2], d[0], d[1]))