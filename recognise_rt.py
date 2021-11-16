import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys

import time
import numpy as np

from mss import mss
from PIL import Image

import tensorflow as tf
from tensorflow import keras

import cv2
from analyze import load_image, find_table, cut_and_warp

# http://www.flyordie.hu/snooker/

labels = ["black", "blue", "brown", "green", "pink", "red", "white", "yellow"]
model = keras.models.load_model("classifier.h5")

bounding_box = {'top': 100, 'left': 100, 'width': 900, 'height': 450}
sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
    img_orig = cv2.resize(img, (800, 450))

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
                    cv2.rectangle(out, rects[i][0], rects[i][1], (0, 0, 255), 2)
                    cv2.putText(out, labels[np.argmax(pred[i])], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                pred = np.array([])
            cv2.imshow("screen", cv2.resize(out, (900, 450)))
        else:
            cv2.imshow("screen", cv2.resize(out, (900, 450)))
    else:
        cv2.imshow("screen", img_orig)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
