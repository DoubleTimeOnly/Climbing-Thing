from typing import List

import cv2
import numpy as np


def imshow(window_name: str, image: np.ndarray, scale: float = 1.0, delay=100):
    output = image.copy()
    if scale != 1.0:
        output = cv2.resize(output, dsize=(0, 0), fx=scale, fy=scale)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, output)

    while delay >= 0:
        k = cv2.waitKey(delay)
        if k != -1 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break     


def mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))


def hstack(images: List[np.ndarray]):
    processed_images = []
    for image in images:
        if image.dtype == np.float32:
            image = (255 * image).astype(np.uint8)
        if len(image.shape) == 2 or image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        processed_images.append(image)
    return np.hstack(processed_images)


def float_to_int(array: np.ndarray):
    return (255 * array).astype(np.uint8)
