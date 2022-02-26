from sklearn_extra.cluster import KMedoids
from sklearn.cluster import *
import cv2
import numpy as np


class Clusterer:
    def __init__(self):
        self.clusterer = None
        self.centers = None
        self.labels = None
        self.fit_data = None

    def fit(self, data):
        try:
            self.clusterer.fit(data)
            self.fit_data = data
            self.centers = self.clusterer.cluster_centers_
            self.labels = self.clusterer.labels_
        except ValueError:
            self.fit_data = data
            self.centers = []
            self.labels = []

    def draw_clusters(self, canvas):
        colors = {}
        for label, point in zip(self.labels, self.fit_data):
            if label == -1:
                color = [0, 0, 0]
            elif label not in colors:
                color = (np.random.random(3) * 255).astype(int).tolist()
                colors[label] = color
            canvas = cv2.circle(canvas, (int(point[0]), int(point[1])), radius=5, color=colors[label], thickness=-1)
        return canvas

    def draw_cluster_centers(self, canvas):
        for point in self.centers:
            color = (np.random.random(3) * 255).astype(int).tolist()
            canvas = cv2.circle(canvas, (int(point[0]), int(point[1])), radius=5, color=color, thickness=-1)
        return canvas


class KMedoids_Clusterer(Clusterer):
    def fit(self, data, n_clusters=1, init="heuristic"):
        self.clusterer = KMedoids(n_clusters=n_clusters,
                                  init=init)
        super().fit(data)


class KMeans_Clusterer(Clusterer):
    def fit(self, data, n_clusters=1, init="k-means++"):
        self.clusterer = KMeans(n_clusters=n_clusters,
                                init=init,
                                max_iter=500)
        super().fit(data)


class DBSCAN_Clusterer(Clusterer):
    def fit(self, data, n_clusters=1, init=None):
        self.clusterer = DBSCAN(eps=100, n_jobs=-1)
        try:
            self.clusterer.fit(data)

            labels = self.clusterer.labels_

            clean_data = []
            for label, point in zip(labels, data):
                if label != -1:
                    clean_data.append(point)

            actual_clusterer = KMeans_Clusterer()
            actual_clusterer.fit(clean_data, n_clusters=n_clusters)
            self.fit_data = data
            self.labels = actual_clusterer.labels
            self.centers = actual_clusterer.centers

        except (ValueError, AttributeError):
            self.fit_data = data
            self.centers = []
            self.labels = []






