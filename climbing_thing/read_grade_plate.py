import numpy as np
import pytesseract
import cv2
import os
import climbing_thing.utils.image as imutils
import re

def main(image_path):
    image = cv2.imread(image_path)

    output = pytesseract.image_to_string(
        image,
        lang='eng',
        config="--psm 7 -c tessedit_char_whitelist=0123456789V --oem 3"
    )
    print(output)
    matches = re.findall(r"(V(B|\d{1,2}))", output)
    for match in matches:
        print(match)
    imutils.imshow("image", image, scale=1)


def main2(image_path):
    image = cv2.imread(image_path)
    image = change_brightness_contrast(image, alpha=1.75, beta=0)
    image = cv2.resize(image, dsize=(0, 0), fx=4, fy=4)
    find_contours(image)
    # image = cv2.GaussianBlur(image, (11, 11), 0)


def change_brightness_contrast(image, alpha=1.0, beta=0):
    image = image.astype(np.float32)
    image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return image


def find_contours(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]

    edges = cv2.Canny(gray, 50, 100, L2gradient=True)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = cv2.drawContours(image, contours, -1, (0, 255, 0),3, lineType=cv2.LINE_AA)

    sbs_image = imutils.hstack([gray, edges, contour_image])
    imutils.imshow("image", sbs_image, delay=0)


def be_sad():
    # image = change_brightness_contrast(image, alpha=1.75, beta=0)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # image = cv2.resize(image, dsize=(0, 0), fx=4, fy=4)

    # image = cv2.GaussianBlur(image, (3, 3), 0)
    # image = cv2.resize(image, dsize=(0, 0), fx=4, fy=4)

    thresh = 250
    # lower_thresh = np.array([0, 0, 0])
    # upper_thresh = np.array([thresh, thresh, thresh])
    # mask = cv2.inRange(image, lower_thresh, upper_thresh)

    # low_hsv_thresh = np.array([0, 10, 0])
    # high_hsv_thresh = np.array([180, 255, 245])
    # mask = cv2.inRange(image, low_hsv_thresh, high_hsv_thresh)

    # imutils.imshow("mask", mask, scale=1, delay=-1)
    # image = cv2.bitwise_and(image, image, mask=mask)
    # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # imutils.imshow("grayscale", image, scale=1, delay=-1)

    # threshold = 0
    # _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    # ret, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print(f"Otsu threshold: {ret}")

if __name__ == "__main__":
    image_dir = r"C:\Users\Victor\Documents\Projects\ClimbingThing\climbing_thing\data\route_grade"
    image_file = "tag_orange_blue_white2.jpg"
    image_file = "test6.jpg"
    image_file = "tag4_all.png"
    # image_file = "test_pink_green2.jpg"
    image_path = os.path.join(image_dir, image_file)
    main(image_path)
    # main2(image_path)