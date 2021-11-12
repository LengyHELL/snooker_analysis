import os
import sys

import math
import numpy as np

import matplotlib.pyplot as plt
import cv2

def load_image(location):

    image = cv2.imread(location)
    if image is None:
        print("Image cannot be read:", "\'" + location + "\'")
        sys.exit(2)
    return image

def intersection(o1, p1, o2, p2):
    x = [o2[0] - o1[0], o2[1] - o1[1]]
    d1 = [p1[0] - o1[0], p1[1] - o1[1]]
    d2 = [p2[0] - o2[0], p2[1] - o2[1]]

    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if (abs(cross) < 1e-8):
        return False

    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    r = [o1[0] + d1[0] * t1, o1[1] + d1[1] * t1]
    return r

def contour_to_quad(contour):
    hull = cv2.convexHull(contour)
    test = [h[0] for h in hull]
    test2 = [[x, x + y, y, x - y, -x, -x - y, -y, -x + y]  for x, y in test]
    points = [0, 0, 0, 0, 0, 0, 0, 0]
    maxes = test2[0]
    #(maximize x, x+y, y, x-y, -x, -x-y, -y, -x+y)
    for i in range(len(test2)):
        for j in range(8):
            if test2[i][j] > maxes[j]:
                points[j] = i
                maxes[j] = test2[i][j]

    oct = []
    for i in range(len(test)):
        if i in points:
            oct.append(test[i])

    lengths = []
    points = []
    for i in range(len(oct)):
        if i == (len(oct) - 1):
            j = 0
        else:
            j = i + 1
        a = oct[j][0] - oct[i][0]
        b = oct[j][1] - oct[i][1]
        lengths.append(math.sqrt(a * a + b * b))
        points.append([oct[i], oct[j]])

    points = np.array(points)
    while len(lengths) > 4:
        rm = min(lengths)
        points = np.delete(points, lengths.index(rm), 0)
        lengths.remove(rm)

    final = []
    for i in range(len(points)):
        if i == (len(points) - 1):
            j = 0
        else:
            j = i + 1
        o1 = points[i][0]
        p1 = points[i][1]
        o2 = points[j][0]
        p2 = points[j][1]

        final.append(intersection(o1, p1, o2, p2))

    return np.array(final, dtype=np.int32)

def find_table(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 190, 50])
    upper_green = np.array([65, 255, 225])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask = mask)

    image_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    result = cv2.Canny(image_gray, 1.0 * threshold, 0.5 * threshold)

    kernel = np.array([ [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0]], dtype=np.uint8)

    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations = 1)
    kernel = np.array([ [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]], dtype=np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x : cv2.contourArea(x), reverse=True)[:1]
    return contour_to_quad(contours[0])

def cut_and_warp(image, contour, size):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1]) **2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1]) **2))
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1]) **2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1]) **2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return cv2.resize(warp, size)

# http://www.flyordie.hu/snooker/

img = load_image("out2.png")
cnt = find_table(img)
img = cut_and_warp(img, cnt, (240, 120))
#img = cv2.drawContours(img, [cnt], -1, (255, 0, 0), 5)


plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()
