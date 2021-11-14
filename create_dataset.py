import os
import time
import matplotlib.pyplot as plt

import cv2
from analyze import load_image, find_table, cut_and_warp, search_template

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
    black   = search_template(img, tmp, threshold=0.855, min_diff=15) #ok
    #tmp = load_image("misc/templates/blue_ball_hd.png")
    #blue    = search_template(img, tmp, threshold=0.92, min_diff=15) #ok
    #tmp = load_image("misc/templates/brown_ball_hd.png")
    #brown   = search_template(img, tmp, threshold=0.92, min_diff=15) #not ok
    #tmp = load_image("misc/templates/green_ball_hd.png")
    #green   = search_template(img, tmp, threshold=0.92, min_diff=15)
    #tmp = load_image("misc/templates/pink_ball_hd.png")
    #pink    = search_template(img, tmp, threshold=0.92, min_diff=15)
    #tmp = load_image("misc/templates/red_ball_hd.png")
    #red     = search_template(img, tmp, threshold=0.92, min_diff=15) #ok
    #tmp = load_image("misc/templates/white_ball_hd.png")
    #white   = search_template(img, tmp, threshold=0.92, min_diff=15)
    #tmp = load_image("misc/templates/yellow_ball_hd.png")
    #yellow  = search_template(img, tmp, threshold=0.92, min_diff=15)

    points = black
    for p in points:
        p = (p[0] - 10, p[1] - 10)
        cv2.rectangle(img, p, (p[0] + 20, p[1] + 20), (0, 0, 255), 2)

    print("(%3d/%3d) %10s - %3d" % (index, len(files), f, len(points)))
    index += 1

    plt.figure(figsize=(16, 6))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
