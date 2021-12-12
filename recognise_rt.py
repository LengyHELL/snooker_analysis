import time
import numpy as np

from mss import mss

import cv2
from analyze import find_table, cut_and_warp, find_circles, load_model, cut_circles, label_cuts_nn

circle_radius = 9
labels = ["black", "blue", "brown", "green", "pink", "red", "white", "yellow"]
model = load_model("classifier_combined3.h5")

bounding_box = {'top': 100, 'left': 100, 'width': 900, 'height': 450}
sct = mss()

while True:
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

        cuts, rects = cut_circles(img, circles, circle_radius, mode="combined")

        ids, _ = label_cuts_nn(cuts, model)
        if ids is None:
            raise Exception("Failed to label candidates!")

        for i, id in enumerate(ids):
            x, y = rects[i][0]
            cv2.rectangle(out, rects[i][0], rects[i][1], (0, 0, 255), 2)
            cv2.putText(out, labels[id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            #cv2.circle(out, (x + 9, y + 9), 10, (0, 0, 255), 2)

    except Exception as e:
        cv2.imshow("screen", cv2.resize(img, (900, 450)))
    else:
        cv2.imshow("screen", cv2.resize(out, (900, 450)))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break