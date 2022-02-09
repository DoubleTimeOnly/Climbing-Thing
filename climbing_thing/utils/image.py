import cv2
import numpy as np


def imshow(window_name: str, image: np.ndarray, scale: float = 1.0, delay=0):
    output = image.copy()
    if scale != 1.0:
        output = cv2.resize(output, dsize=(0, 0), fx=scale, fy=scale)
    cv2.imshow(window_name, output)
    if delay >= 0:
        cv2.waitKey(delay)

def mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))