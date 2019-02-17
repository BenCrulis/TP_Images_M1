import numpy as np
import cv2
import os


jimmy = "/home/alex/PycharmProjects/tp_img_processing/data/jimmy_fallon.mp4"
save_folder = "/home/alex/PycharmProjects/tp_img_processing/github/plans_sift"

cap = cv2.VideoCapture(jimmy)


def greyscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(g.shape)
    return g

# Sift "distance"
sift = cv2.xfeatures2d.SIFT_create()
def similarity_sift(img1, img2):
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return len(good) / float(len(kp2))

def representant(images: list):
    n = len(images)
    distances = np.zeros(shape=[n, n], dtype=np.float32)
    print("Computing distance Matrix")
    for i in range(n):
        for j in range(i + 1):
            d = 1 - similarity_sift(images[i], images[j])
            distances[i, j] = d
            distances[j, i] = d
    print("Distances compute complete")
    costs = distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)

i = 0
limit = 1500
plans = [[]]
current = 0
prev = None

for i in range(96):
    i += 1
    cap.read()

while True:
    i += 1
    ret, frame = cap.read()
    if frame is None or i > limit:
        break

    if ret:
        img = frame.copy()
        img = greyscale(img)

        if prev is not None:
            if similarity_sift(img, prev) < 0.3:
                print("Changement de Plan  à l'image {}".format(i))
                if not os.path.exists(save_folder + "/plan_" + str(current)):
                    os.mkdir(save_folder + "/plan_" + str(current))
                plans.append([])
                current += 1
                plans[current].append(img)
            else:
                plans[current].append(img)
            cv2.imwrite(save_folder + "/plan_" + str(current) + "/" + str(i) + ".png", img)

        prev = img
    else:
        print('video ended')
        break

print("detected", len(plans), " plans")
cap.release()
