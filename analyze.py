import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys

import math
import numpy as np

import cv2
import tensorflow as tf
from tensorflow import keras


def load_image(location):

    image = cv2.imread(location, cv2.IMREAD_COLOR)
    if image is None:
        print("Image cannot be read:", "\'" + location + "\'")
        sys.exit(2)
    return image

def intersection(p1, p2, p3, p4):
    d = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    if (abs(d) < 1e-8):
        return False
    
    t1 = (p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p3[0] * p4[1] - p3[1] * p4[0]) * (p1[0] - p2[0])
    t2 = (p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p3[0] * p4[1] - p3[1] * p4[0]) * (p1[1] - p2[1])
    return [t1 / d, t2 / d]

def get_length(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def contour_to_quad(contour):
    coords = np.array([c[0] for c in contour])

    lengths = []
    for i in range(len(coords)):
        lengths.append([get_length(coords[i-1], coords[i]), coords[i - 1], coords[i]])

    lengths = sorted(lengths, key=lambda x : x[0], reverse=True)[:4]
    temp = lengths[1]
    lengths[1] = lengths[2]
    lengths[2] = temp

    points = []
    for i in range(len(lengths)):
        _, o1, p1 = lengths[i - 1]
        _, o2, p2 = lengths[i]
        point = intersection(o1, p1, o2, p2)
        if point == False:
            return None
        if (point[0] < 0) or (point[1] < 0):
            return None
        points.append(point)

    points = np.array(points)
    points = np.int0(points)
    return points.reshape(4, 1, 2)

def find_table(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 190, 50])
    upper_green = np.array([65, 255, 225])
    #lower_green = np.array([30, 180, 40])
    #upper_green = np.array([75, 255, 235])
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

    contours = [cv2.convexHull(c) for c in contours]
    #for i, c in enumerate(contours):
    #    hull = cv2.convexHull(c)
    #    epsilon = 0.005 * cv2.arcLength(hull, True)
    #    contours[i] = cv2.approxPolyDP(hull, epsilon, True)
    contours = sorted(contours, key=lambda x : cv2.contourArea(x), reverse=True)[:1]

    if len(contours) <= 0:
        return None

    cnt = contours[0]

    quad = contour_to_quad(cnt)
    if quad is None:
        return None
    else:
        return quad

def cut_and_warp(image, contour, size):
    pts = contour.reshape(4, 2)
    src = np.zeros((4, 2), "float32")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).flatten()

    #arranging coords to: tl tr br bl
    src[0] = pts[np.argmin(sums)]
    src[1] = pts[np.argmin(diffs)]
    src[2] = pts[np.argmax(sums)]
    src[3] = pts[np.argmax(diffs)]

    width, height = (size[0] - 1, size[1] - 1)

    dst = np.array([
        [0,     0       ],
        [width, 0       ],
        [width, height  ],
        [0,     height  ]],
        dtype="float32")
    
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(image, M, size)
    return warp

def calculate_diff(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def search_template(image, template, threshold=0.9, min_diff=15):
    w, h, _ = template.shape

    res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    loc = np.where(res >= threshold)

    #print(np.array(loc))
    points = []
    for pt in zip(*loc[::-1]):
        pt = (int(pt[0] + w/2), int(pt[1] + h/2))
        add = True
        for p in points:
            diff = calculate_diff(pt, p)
            if diff < min_diff:
                add = False
        if add:
            points.append(pt)
    return np.array(points)

def load_model(path):
    return keras.models.load_model(path)

def find_circles(image, circle_radius):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0] * 0.8
    if threshold <= 0:
        return None

    minRadius = int(circle_radius * 0.6)
    maxRadius = int(circle_radius * 1.2)
    minDistance = int(circle_radius * 1.6)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDistance, param1=threshold, param2=15, minRadius=minRadius, maxRadius=maxRadius)
    if circles is None:
        return None
    
    return np.round(circles[0,:]).astype("int")

def cut_circles(img, circles, circle_radius, mode="bgr"):
    cuts = []
    rects = []
    hsv = None
    if mode != "bgr":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    for (x, y, r) in circles:
        x, y, w, h = (x-circle_radius, y-circle_radius, circle_radius * 2, circle_radius * 2)
        if (x >= 0) and (y >= 0) and ((x + w) < img.shape[1]) and ((y + h) < img.shape[0]):
            if mode == "bgr":
                cuts.append(img[y:y+h, x:x+w])
            elif mode == "hsv":
                cuts.append(hsv[y:y+h, x:x+w])
            elif mode == "combined":
                cuts.append(np.append(img[y:y+h, x:x+w], hsv[y:y+h, x:x+w], axis=2))
            else:
                return None, None
            rects.append(((x, y), (x+w, y+h)))
    return np.array(cuts), rects

def label_cuts_tm(cuts, templates):
    labels = []
    preds = []
    for c in cuts:
        res = []
        for t in templates:
            res.append(cv2.matchTemplate(c, t, cv2.TM_CCORR_NORMED))
        res = np.array(res).reshape(-1)
        preds.append(res)
        labels.append(np.argmax(res))
    return np.array(labels), np.array(preds)
            
def label_cuts_nn(cuts, model):
    norms = np.array(cuts / 255)
    if len(norms) > 0:
        preds = model.predict(norms)
        labels = np.argmax(preds, axis=1)
    else:
        return None, None
    return labels, preds