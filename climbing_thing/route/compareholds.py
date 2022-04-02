from math import fabs
from typing import Dict, List

import torch

from climbing_thing.climbnet import Instances
import cv2
import numpy as np
from scipy.spatial import distance

from climbing_thing.utils.distancemetrics import (
    compute_hsv_histogram,
    l2_norm,
    l1_norm,
    linf_norm,
    cosine_similarity,
    jensenshannon,
    wasserstein_distance
)
import matplotlib.pyplot as plt

DISTANCES = {
    'l2_norm': l2_norm,
    'l1_norm': l1_norm,
    'linf_norm': linf_norm,
    'cosine similarity': cosine_similarity,
    'jenssen shannon': jensenshannon,
    'wasserstein': wasserstein_distance,
}

def get_masked_image(image: np.ndarray, mask: torch.tensor):
    mask = np.array(mask.long()).astype(np.uint8)
    masked_image = image[mask > 0]
    return masked_image


def write_csv(csv_list: List, distance_name: str):
    with open(f"cartesian_distances_{distance_name}.csv", 'w') as csv_file:
        text = "\n".join(csv_list)
        csv_file.write(text)



def compute_cartesian_difference(route_image: np.ndarray, holds: Instances):
    bins_per_channel = 18
    hsv_image = cv2.cvtColor(route_image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("Controller", np.random.random((200, 200)))

    def hold_to_feature_vector(mask):
        mask = mask.to("cpu")
        mask = np.array(mask.long()).astype(np.uint8)
        (h_hist, h_edges), (s_hist, s_edges), (v_hist, v_edges) = compute_hsv_histogram(hsv_image, bins=bins_per_channel, mask=mask, mode="np")
        return np.concatenate([h_hist, s_hist, v_hist], axis=0)

    feature_vectors = np.array([hold_to_feature_vector(mask) for mask in holds.masks])
    return distance.squareform(distance.pdist(feature_vectors, 'minkowski', p=2))
    # csv_dict = {metric: [] for metric in DISTANCES}

    # for histogram in histograms:
    #     for histogram in histograms:
    #       for name, dist_func in DISTANCES.items():
                

    # for idx, mask in enumerate(holds.masks):
    #     print(f"Outer Hold: {idx}")
    #     mask = mask.to("cpu")
    #     mask = np.array(mask.long()).astype(np.uint8)

    #     (h_hist, h_edges), (s_hist, s_edges), (v_hist, v_edges) = compute_hsv_histogram(hsv_image, bins=bins_per_channel, mask=mask, mode="np")

    #     feature_vector1 = np.concatenate([h_hist, s_hist, v_hist], axis=0)

    #     all_distances = {}



    #     for inner_idx, mask2 in enumerate(holds.masks):
    #         # print(f"Hold: {inner_idx}")
    #         distances = {}
    #         mask2 = mask2.to("cpu")
    #         mask2 = np.array(mask2.long()).astype(np.uint8)

    #         (h_hist2, h_edges2), (s_hist2, s_edges2), (v_hist2, v_edges2) = compute_hsv_histogram(hsv_image, bins=bins_per_channel, mask=mask2, mode="np")
    #         feature_vector2 = np.concatenate([h_hist2, s_hist2, v_hist2], axis=0)


    #         for metric in DISTANCES:
    #             distances[metric] = DISTANCES[metric](feature_vector1, feature_vector2)
    #             # print(f"\t{metric}: {distances[metric]:.2f}")

    #         all_distances[inner_idx] = distances

    #         # Plots are cool
    #         # scale = 1.2
    #         # fig, ax = plt.subplots(2, 3, figsize=(12.8*scale, 9.6*scale))
    #         # hold_bbox = holds.boxes[idx].tensor.int()
    #         # hold_bbox2 = holds.boxes[inner_idx].tensor.int()

    #         # ax[0, 0].imshow(route_image[..., ::-1][hold_bbox[0, 1]:hold_bbox[0, 3], hold_bbox[0, 0]:hold_bbox[0, 2]])
    #         # ax[0, 1].imshow(route_image[..., ::-1][hold_bbox2[0, 1]:hold_bbox2[0, 3], hold_bbox2[0, 0]:hold_bbox2[0, 2]])
    #         #
    #         # ax[1, 0].stairs(h_hist, h_edges, color='b', ls="--")
    #         # ax[1, 0].stairs(h_hist2, h_edges2, color='b')
    #         # ax[1, 0].set_ylim(0, 0.4)
    #         #
    #         # ax[1, 1].stairs(s_hist, s_edges, color='g', ls="--")
    #         # ax[1, 1].stairs(s_hist2, s_edges2, color='g')
    #         # ax[1, 1].set_ylim(0, 0.4)
    #         #
    #         # ax[1, 2].stairs(v_hist2, v_edges2, color='r', ls="--")
    #         # ax[1, 2].stairs(v_hist, v_edges, color='r')
    #         # ax[1, 2].set_ylim(0, 0.4)
    #         #
    #         # ax[0, 2].axis('off')
    #         #
    #         # plt.tight_layout()
    #         # plt.show(block=True)
    #         #
    #         # key = cv2.waitKey(0)
    #         # if key == ord("s"):
    #         #     break
    #         #
    #         # plt.close(fig)


    #     for metric in DISTANCES:
    #         csv = get_csv(all_distances, key=metric)
    #         csv_dict[metric].append(csv)

    # for metric, csv in csv_dict.items():
    #     write_csv(csv, distance_name=metric)


def get_csv(distances: Dict, key):
    csv = ""
    for value in distances.values():
        value = value[key]
        csv += f"{value},"

    return csv.strip(",")


