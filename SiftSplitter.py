import numpy as np
import cv2
import os
import random as rd

jimmy = "jimmy_fallon.mp4"
save_folder = "./plans_sift"
save = True

print("starting capture")

cap = cv2.VideoCapture(jimmy)


def greyscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(g.shape)
    return g

# Sift "distance"
sift = cv2.xfeatures2d.SIFT_create(16)
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
    return costs.argmin(axis=0)

def representant_greedy(images: list, n_iter: int, sampling_fraction: float):
    
    best = rd.choice(images)
    best_score = 0.0
    
    for i in range(n_iter):
        elected = rd.choice(images)
                
        scores = [similarity_sift(img[0], elected[0]) for img in
            rd.choices(images, k=int(sampling_fraction*len(images)))]
        
        sc = sum(scores)/len(scores)
        
        if sc > best_score:
            best_score = sc
            best = elected
    
    return (best[0], best[1], best_score)
        

i = 0
limit = 1500
plans = [[]]
frames = []
current = 0
if save and not os.path.exists(save_folder + "/plan_" + str(current)):
    os.mkdir(save_folder + "/plan_" + str(current))
prev = None

print("skipping a few frames")

for i in range(96):
    ret, frame = cap.read()
    frames.append(frame)
    i += 1


print("beginning main loop at frame {}".format(i))

while True:    
    if i % 100 == 0:
        print("frame {}".format(i))
    
    ret, frame = cap.read()
    if frame is None or i > limit:
        break

    if ret:
        frames.append(frame)
        img = frame.copy()
        img = greyscale(img)

        cv2.imshow("video", frame)

        if prev is not None:
            if similarity_sift(img, prev) < 0.15:
                print("Changement de Plan  à l'image {}".format(i))
                if save and not os.path.exists(save_folder + "/plan_" + str(current)):
                    os.mkdir(save_folder + "/plan_" + str(current))
                plans.append([])
                current += 1
            plans[current].append((img,i))
            if save:
                cv2.imwrite(save_folder + "/plan_" + str(current) + "/" + str(i) + ".png", img)
        prev = img
    else:
        print('video ended')
        break
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
    i += 1


print("detected", len(plans), " plans")
cap.release()

for i,plan in enumerate(plans):
    rep, ind, score = representant_greedy(plan, 10, 0.05)
    print("found representant with score {} for plan n°{}: image {}".format(score,i+1, ind))
    cv2.imshow("representent", frame)
    
    if save:
        cv2.imwrite(save_folder + "/resume_{}_score_{:.2f}_.png".format(ind, score), frames[ind])
    
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
