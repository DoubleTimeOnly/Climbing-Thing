from typing import Dict, List, Union
from math import fabs

import scipy
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from scipy import stats
from scipy.spatial.distance import pdist

from climbing_thing.climbnet import Instances
from climbing_thing.utils.distancemetrics import compute_histograms, l2_norm, l1_norm, linf_norm, cosine_similarity

DISTANCES = {
    'l1_norm': {"metric": "minkowski", "p": 1},
    'l2_norm': {"metric": "minkowski", "p": 2},
    'cosine similarity': {"metric": "cosine"},
    'jenssen shannon': {"metric": "jensenshannon"},
    'wasserstein': {"metric": stats.wasserstein_distance},
}


def get_masked_image(image: np.ndarray, mask: torch.tensor):
    mask = np.array(mask.long()).astype(np.uint8)
    masked_image = image[mask > 0]
    return masked_image


def write_csv(csv_list: List, distance_name: str):
    with open(f"cartesian_distances_{distance_name}.csv", 'w') as csv_file:
        text = "\n".join(csv_list)
        csv_file.write(text)


def compute_cartesian_difference(route_image: np.ndarray, holds: Instances, color_space="hsv"):
    color_spaces = {
        "hsv": {"conversion": cv2.COLOR_BGR2HSV, "bins": 180, "max_values": [180, 256, 256]},
        "hsv_bin_accurate": {"conversion": cv2.COLOR_BGR2HSV, "bins": [180, 256, 256], "max_values": [180, 256, 256]},
        "hsv_256": {"conversion": cv2.COLOR_BGR2HSV, "bins": 256, "max_values": [256, 256, 256]},
        "lab": {"conversion": cv2.COLOR_BGR2LAB, "bins": 256, "max_values": [256, 256, 256]},
        "bgr": {"conversion": None, "bins": 256, "max_values": [256, 256, 256]},
        "hls": {"conversion": cv2.COLOR_BGR2HLS, "bins": 180, "max_values": [180, 256, 256]},
    }

    params = color_spaces[color_space]

    if params["conversion"] is not None:
        image = cv2.cvtColor(route_image, params["conversion"])
    else:
        image = route_image

    def hold_to_feature_vector(mask):
        mask = mask.to("cpu")
        mask = np.array(mask.long()).astype(np.uint8)
        output = compute_histograms(image, bins=params["bins"], mask=mask, mode="np", max_values=params["max_values"])
        (h_hist, h_edges), (s_hist, s_edges), (v_hist, v_edges) = output
        n = 11
        kernel = [1/n for i in range(n)]
        h_hist = np.convolve(h_hist, kernel, mode="same")
        s_hist = np.convolve(s_hist, kernel, mode="same")
        v_hist = np.convolve(v_hist, kernel, mode="same")
        return np.concatenate([h_hist, s_hist, v_hist], axis=0)

    feature_vectors = np.array([hold_to_feature_vector(mask) for mask in holds.masks])

    computed_distances = {}
    for metric, kwargs in DISTANCES.items():
        distances = pdist(feature_vectors, **kwargs)
        computed_distances[metric] = distance.squareform(distances)
    return computed_distances
