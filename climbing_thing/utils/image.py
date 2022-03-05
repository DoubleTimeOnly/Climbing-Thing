import cv2
import numpy as np


def imshow(window_name: str, image: np.ndarray, scale: float = 1.0, delay=100):
    output = image.copy()
    if scale != 1.0:
        output = cv2.resize(output, dsize=(0, 0), fx=scale, fy=scale)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, output)
    if delay < 1:
        delay = 100
    while True:
        k = cv2.waitKey(100)
        if k != -1:
            print("Pressed Key Code", k)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:        
            break     

def mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
