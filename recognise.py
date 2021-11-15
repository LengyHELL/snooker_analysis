import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import cv2
from analyze import load_image, find_table, cut_and_warp

# http://www.flyordie.hu/snooker/

dir = "./misc/images"
files = os.listdir(dir)
index = 1

labels = ["black", "blue", "brown", "green", "pink", "red", "white", "yellow"]
model = keras.models.load_model("classifier.h5")

for f in files:
    start_time = time.time()
    img = load_image(dir + "/" + f)
    cnt = find_table(img)
    img = cut_and_warp(img, cnt, (1024, 512))

    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 15, param2=15, minRadius=5, maxRadius=11)

    cuts = []
    cut = img.copy()
    if circles is not None:
        circles = np.round(circles[0,:]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(out, (x, y), r, (255, 0, 0), 1)
            x = x - 9
            y = y - 9
            w = 18
            h = 18
            cuts.append(img[y:y+h, x:x+w])
    cuts = np.array(cuts)
    norms = np.array(cuts / 255)
    pred = model.predict(norms)

    print("%-27s -> | %d ms" % ("Total", (time.time() - start_time) * 1000))

    #for i in range(len(cuts)):
    #    plt.figure(figsize=(10, 5))
    #    print(labels[np.argmax(pred[i])])
    #    plt.imshow(cv2.cvtColor(cuts[i], cv2.COLOR_BGR2RGB))
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.show()
