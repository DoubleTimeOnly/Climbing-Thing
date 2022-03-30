from typing import Dict, List, Union

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import math

from climbing_thing.climbnet import Instances
from climbing_thing.utils.distancemetrics import compute_hsv_histogram, l2_norm, l1_norm, linf_norm, cosine_similarity
import logging

DISTANCES = {
    'l1_norm': {"metric": "minkowski", "p": 1},
    'l2_norm': {"metric": "minkowski", "p": 2},
    # 'linf_norm': linf_norm,
    'cosine similarity': {"metric": "cosine"},
}


class DistanceMatrix:
    def __init__(self, distances: np.ndarray, num_observations: int):
        """
        :param distances: distances from scipy pdist
        """
        self.distances = distances
        self.num_observations = num_observations

    def get_item(self, row: int, col: Union[int, List]) -> Union[int, List]:
        """
        Return the distance between distances[row] and distances[col]
        :param row, col: indices of entries to get distance of. Col can be a list of indices to return a list of distances
        """
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        if isinstance(col, int):
            return self.unroll_entry(row, col)

        elif isinstance(col, List):
            return [self.unroll_entry(row, c) for c in col]

    def unroll_entry(self, row: int, col: int):
        if row >= self.num_observations or col >= self.num_observations:
            raise IndexError(f"Invalid indices {row, col} for matrix of length {len(self.distances)}")
        if row == col:
            # logging.warning(f"Warning: pdist does not calculate distance between the same vertices. Returning 0 by default")
            return 0
        i = min(row, col) if row != col else row
        j = max(row, col) if row != col else col
        entry = self.num_observations * i + j - ((i + 2) * (i + 1)) // 2
        return self.distances[entry]


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
        "hsv": {"conversion": cv2.COLOR_BGR2HSV, "bins": 180},
        "lab": {"conversion": cv2.COLOR_BGR2LAB, "bins": 256},
    }
    params = color_spaces[color_space]
    image = route_image#.astype(np.float32)
    hsv_image = cv2.cvtColor(image, params["conversion"])
    # cv2.imshow("Controller", np.random.random((200, 200)))

    # csv_dict = {metric: [] for metric in DISTANCES}
    computed_distances = {}

    hold_histograms = []
    for idx, mask in enumerate(holds.masks):
        mask = mask.to("cpu")
        mask = np.array(mask.long()).astype(np.uint8)

        output = compute_hsv_histogram(hsv_image, bins=params["bins"], mask=mask, mode="np", max_values=color_space)
        (h_hist, h_edges), (s_hist, s_edges), (v_hist, v_edges) = output
        feature_vector1 = np.concatenate([h_hist, s_hist, v_hist], axis=0)
        hold_histograms.append(feature_vector1)
    hold_histograms = np.array(hold_histograms, dtype=np.float32)

    for metric, kwargs in DISTANCES.items():
        distances = pdist(hold_histograms, **kwargs)
        computed_distances[metric] = DistanceMatrix(distances, len(hold_histograms))

    return computed_distances




"""
        for inner_idx, mask2 in enumerate(holds.masks):
            # print(f"Hold: {inner_idx}")
            distances = {}
            mask2 = mask2.to("cpu")
            mask2 = np.array(mask2.long()).astype(np.uint8)

            (h_hist2, h_edges2), (s_hist2, s_edges2), (v_hist2, v_edges2) = compute_hsv_histogram(hsv_image, bins=180, mask=mask2, mode="np")
            feature_vector2 = np.concatenate([h_hist2, s_hist2, v_hist2], axis=0)


            for metric in DISTANCES:
                distances[metric] = DISTANCES[metric](feature_vector1, feature_vector2)
                # print(f"\t{metric}: {distances[metric]:.2f}")

            all_distances[inner_idx] = distances

            # Plots are cool
            scale = 1.2
            # fig, ax = plt.subplots(2, 3, figsize=(12.8*scale, 9.6*scale))
            # hold_bbox = holds.boxes[idx].tensor.int()
            # hold_bbox2 = holds.boxes[inner_idx].tensor.int()

            # ax[0, 0].imshow(route_image[..., ::-1][hold_bbox[0, 1]:hold_bbox[0, 3], hold_bbox[0, 0]:hold_bbox[0, 2]])
            # ax[0, 1].imshow(route_image[..., ::-1][hold_bbox2[0, 1]:hold_bbox2[0, 3], hold_bbox2[0, 0]:hold_bbox2[0, 2]])
            #
            # ax[1, 0].stairs(h_hist, h_edges, color='b', ls="--")
            # ax[1, 0].stairs(h_hist2, h_edges2, color='b')
            # ax[1, 0].set_ylim(0, 0.4)
            #
            # ax[1, 1].stairs(s_hist, s_edges, color='g', ls="--")
            # ax[1, 1].stairs(s_hist2, s_edges2, color='g')
            # ax[1, 1].set_ylim(0, 0.4)
            #
            # ax[1, 2].stairs(v_hist2, v_edges2, color='r', ls="--")
            # ax[1, 2].stairs(v_hist, v_edges, color='r')
            # ax[1, 2].set_ylim(0, 0.4)
            #
            # ax[0, 2].axis('off')
            #
            # plt.tight_layout()
            # plt.show(block=True)
            #
            # key = cv2.waitKey(0)
            # if key == ord("s"):
            #     break
            #
            # plt.close(fig)
"""

        # for metric in DISTANCES:
        #     csv = get_csv(all_distances, key=metric)
        #     csv_dict[metric].append(csv)

    # for metric, csv in csv_dict.items():
    #     write_csv(csv, distance_name=metric)


def get_csv(distances: Dict, key):
    csv = ""
    for value in distances.values():
        value = value[key]
        csv += f"{value},"

    return csv.strip(",")


