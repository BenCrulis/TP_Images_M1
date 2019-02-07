import numpy as np
import cv2
import functools
import os


jimmy = "/home/alex/PycharmProjects/tp_img_processing/data/jimmy_fallon.mp4"
save_folder = "/home/alex/PycharmProjects/tp_img_processing/github/plans"

cap = cv2.VideoCapture(jimmy)


def greyscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(g.shape)
    return g


i = 0
limit = 1500
plans = [[]]
current = 0
prev = None
while True:
    i += 1
    ret, frame = cap.read()
    if frame is None or i > limit:
        break
    max_change = frame.shape[0]*frame.shape[1]
    epsilon = 0.07
    max_change = (1-epsilon)*max_change*127
    if ret:
        img = frame.copy()
        img = greyscale(img)

        res = img
        if prev is not None:
            res = img - greyscale(prev)
            if np.sum(res) > max_change:
                print("Changement de Plan  à l'image {}".format(i))
                if not os.path.exists(save_folder + "/plan_" + str(current)):
                    os.mkdir(save_folder + "/plan_" + str(current))
                plans.append([])
                current += 1
                plans[current].append(img)
            else:
                plans[current].append(img)
            cv2.imwrite(save_folder + "/plan_" + str(current) + "/" + str(i) + ".png", img)

        prev = frame
    else:
        print('video ended')
        break

print("detected", len(plans), " plans")
cap.release()
