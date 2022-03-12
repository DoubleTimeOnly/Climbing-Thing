import os
from climbing_thing.utils.image import imshow
import cv2
import numpy as np

from climbing_thing.patternmatching.patmatch import PatternMatcher


def find_plate(query_path: str, pattern_path: str):
    query = cv2.imread(query_path)
    pattern = cv2.imread(pattern_path, 0)

    # plate_matcher = PatternMatcher(pattern_path)
    # matched_patterns = plate_matcher.find_pattern(query, n_matches=1)
    # print(matched_patterns)
    # cv2.waitKey(0)

    gray_query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray_query, pattern, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    w, h = pattern.shape[::-1]

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(query,top_left, bottom_right, 255, 2)
    imshow("match", query, scale=0.2, delay=0)


if __name__ == '__main__':
    image_dir = r"C:\Users\Victor\Documents\Projects\ClimbingThing\climbing_thing\data\images"
    image_file = "20220207_215052.jpg"
    image_file = "20220207_215101.jpg"
    query_path = os.path.join(image_dir, image_file)
    pattern_path = r"C:\Users\Victor\Documents\Projects\ClimbingThing\climbing_thing\data\templates\grade_plate.jpg"
    find_plate(query_path, pattern_path)

