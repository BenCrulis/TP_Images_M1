import numpy as np
import cv2
import os
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random


def cluster(distances, k=3):
    m = distances.shape[0]  # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1] * k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1] * k)  # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1] * k)

    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids


def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:, medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters


def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster, cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)

def greyscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
            d = 1 - similarity_sift(imgs[i], imgs[j])
            distance[i, j] = d
            distance[j, i] = d
    print("Distances compute complete")
    costs = distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)


folder = "/home/alex/PycharmProjects/tp_img_processing/github/plans/all"

imgs = []
img_map = {}

print("Loading images in memory ...")
for dirpath, dirnames, filnames in os.walk(folder):
    for file in filnames:
        if file.endswith(".png"):
            abspath = os.path.join(dirpath, file)
            img = cv2.imread(abspath, 0)
            img_map[len(img)] = abspath
            imgs.append(img)
print("Loading complete")

n = len(imgs)
distance = np.zeros(shape=[n, n], dtype=np.float32)

for i in range(n):
    for j in range(i + 1):
        d = 1 - similarity_sift(imgs[i], imgs[j])
        distance[i, j] = d
        distance[j, i] = d
    print("Line", i, "done")

pprint(distance)
pprint(img_map)
np.save("distance.np", distance)

