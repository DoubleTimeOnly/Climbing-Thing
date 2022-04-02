import cv2
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


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

def compute_hsv_histogram(hsv_image, bins, mask, mode="cv2"):
    if isinstance(bins, int):
        bins = [bins, bins, bins]

    if mode == "cv2":
        h_hist = cv2.calcHist([hsv_image[..., 0]], [0], mask, [bins[0]], [0, 180])
        s_hist = cv2.calcHist([hsv_image[..., 1]], [0], mask, [bins[1]], [0, 256])
        v_hist = cv2.calcHist([hsv_image[..., 2]], [0], mask, [bins[2]], [0, 256])

        return h_hist, s_hist, v_hist
    elif mode == "np":
        image = hsv_image[[mask > 0]] if mask is not None else hsv_image
        total_value = mask.sum() if mask is not None else np.prod(image.shape[:2])
        h_hist, h_edges = np.histogram(image[..., 0], bins=np.linspace(0, 180, bins[0], endpoint=False))
        s_hist, s_edges = np.histogram(image[..., 1], bins=np.linspace(0, 256, bins[1], endpoint=False))
        v_hist, v_edges = np.histogram(image[..., 2], bins=np.linspace(0, 256, bins[2], endpoint=False))

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
