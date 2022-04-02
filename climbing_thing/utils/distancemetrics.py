from typing import List, Union

import cv2
import numpy as np
from numpy.linalg import norm


def l2_norm(array1: np.ndarray, array2: np.ndarray):
    array = array1 - array2
    return norm(array, ord=2)


def l1_norm(array1: np.ndarray, array2: np.ndarray):
    array = array1 - array2
    return norm(array, ord=1)


def linf_norm(array1: np.ndarray, array2: np.ndarray):
    array = array1 - array2
    return norm(array, ord=np.inf)


def cosine_similarity(array1: np.ndarray, array2: np.ndarray):
    dot_product = np.dot(array1.T, array2)
    similarity = dot_product / (norm(array1, ord=2) * norm(array2, ord=2))
    return similarity


def compute_hsv_histogram(hsv_image, bins, mask, max_values: Union[str, List] = None, mode="cv2"):
    if isinstance(bins, int):
        bins = [bins, bins, bins]

    if isinstance(max_values, str):
        channel_ranges = {
            "hsv": [180, 256, 256],
            "rgb": [256, 256, 256],
            "lab": [256, 256, 256],
        }
        max_values = channel_ranges[max_values]
    elif isinstance(max_values, List):
        max_values = max_values if max_values is not None else [180, 256, 256]
    else:
        raise NotImplementedError(f"Max value range {max_values} not supported")

    if mode == "cv2":
        h_hist = cv2.calcHist([hsv_image[..., 0]], [0], mask, [bins[0]], [0, max_values[0]])
        s_hist = cv2.calcHist([hsv_image[..., 1]], [0], mask, [bins[1]], [0, max_values[1]])
        v_hist = cv2.calcHist([hsv_image[..., 2]], [0], mask, [bins[2]], [0, max_values[2]])

        return h_hist, s_hist, v_hist
    elif mode == "np":
        image = hsv_image[[mask > 0]] if mask is not None else hsv_image
        total_value = mask.sum() if mask is not None else np.prod(image.shape[:2])
        h_hist, h_edges = np.histogram(image[..., 0], bins=np.linspace(0, max_values[0], bins[0], endpoint=False))
        s_hist, s_edges = np.histogram(image[..., 1], bins=np.linspace(0, max_values[1], bins[1], endpoint=False))
        v_hist, v_edges = np.histogram(image[..., 2], bins=np.linspace(0, max_values[2], bins[2], endpoint=False))

        h_hist = h_hist / total_value
        s_hist = s_hist / total_value
        v_hist = v_hist / total_value

        return (h_hist, h_edges), (s_hist, s_edges), (v_hist, v_edges)
    else:
        raise ValueError(f"Unknown mode {mode}")


if __name__ == '__main__':
    array1 = np.array([[0, 1, 0]])
    array2 = np.array([[0, 0, 1]])

    print(l2_norm(array1, array2))
    print(cosine_similarity(array1, array2))

    print(l2_norm(array2, array2))
    print(cosine_similarity(array2, array2))
