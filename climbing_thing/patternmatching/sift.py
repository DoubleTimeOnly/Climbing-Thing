from climbing_thing.patternmatching import matcher
from climbing_thing.utils import logger
import cv2
import numpy as np
from climbing_thing.patternmatching.clusterer import *


log = logger.get_logger(__name__)
log.setLevel(logger.DEBUG_WITH_IMAGES)


class SIFTMatcher(matcher.Matcher):
    def __init__(self):
        super().__init__()
        self.sift = cv2.SIFT_create(nOctaveLayers=3, sigma=3)
        self.clusterer = DBSCAN_Clusterer()

    def find_matches(self, query, pattern, n_matches=1):
        if query is None or pattern is None:
            raise ValueError(f"One of query or pattern is None. Query: {type(query)}. Pattern: {type(pattern)}.")
        outputs = self.detect_keypoints_and_descriptors(query, pattern)
        query_keypoints, query_descriptors = outputs[0]
        pattern_keypoints, pattern_descriptors = outputs[1]

        good_matches, matches_mask, descriptor_matches = self.match_descriptors(query_descriptors, pattern_descriptors)

        cluster_centers, cluster_labels = self.cluster_keypoints(query_keypoints, good_matches, n_clusters=n_matches)

        if log.level <= logger.DEBUG_WITH_IMAGES:
            canvas = query.copy()
            self.clusterer.draw_clusters(canvas)
            self.clusterer.draw_cluster_centers(canvas)
            scale = 0.2
            canvas = cv2.resize(canvas, (0, 0), fx=scale, fy=scale)
            cv2.imshow("keypoint matches", canvas)
            cv2.waitKey(1)
        return cluster_centers

    def match_descriptors(self, query_descriptors, pattern_descriptors):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        descriptor_matches = flann.knnMatch(query_descriptors, pattern_descriptors, k=2)

        good_matches = []
        matchesMask = [[0, 0] for i in range(len(descriptor_matches))]

        # Reduce number of false positives using lowe's ratio test
        for i, (m, n) in enumerate(descriptor_matches):
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)
                matchesMask[i] = [1, 0]

        return good_matches, matchesMask, descriptor_matches

    def detect_keypoints_and_descriptors(self, query, pattern):
        query_keypoints, query_descriptors = self.sift.detectAndCompute(query, None)
        pattern_keypoints, pattern_descriptors = self.sift.detectAndCompute(pattern, None)

        outputs = [
            (query_keypoints, query_descriptors),
            (pattern_keypoints, pattern_descriptors)
        ]
        return outputs

    def cluster_keypoints(self, query_keypoints, good_matches, n_clusters=1):
        query_keypoint_matches = np.array([query_keypoints[match.queryIdx].pt for match in good_matches])
        self.clusterer.fit(query_keypoint_matches, n_clusters=n_clusters)
        return self.clusterer.centers, self.clusterer.labels

    @staticmethod
    def draw_keypoint_matches(matches_mask, query, query_keypoints, pattern, pattern_keypoints, descriptor_matches):
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask,
                           flags=0)
        canvas = cv2.drawMatchesKnn(query, query_keypoints, pattern, pattern_keypoints, descriptor_matches, None, **draw_params)
        return canvas


