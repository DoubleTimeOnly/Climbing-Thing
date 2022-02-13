import os

from climbing_thing.route import Route
from climbing_thing.climbnet import ClimbNet
from climbing_thing.utils.image import imshow
from climbing_thing.utils import logger
import cv2

log = logger.get_logger(__name__)
log.setLevel(logger.DEBUG)


def main(image_dir, mask_dir):
    scale = 0.33
    check_dirs(image_dir, mask_dir)

    image_paths = os.listdir(image_dir)
    image_paths = remove_non_image_paths(image_paths)

    for image_path in image_paths:
        full_image_path = os.path.join(image_dir, image_path)
        route_image = cv2.imread(full_image_path)

        imshow("route", route_image, scale=scale)



def check_dirs(image_dir, mask_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Could not find directory: {os.path.abspath(image_dir)}"
        )

    if not os.path.exists(mask_dir):
        log.debug(f"Creating mask directory: {os.path.abspath(mask_dir)}")
        os.mkdir(mask_dir)


def remove_non_image_paths(image_paths, extensions=None):
    if extensions is None:
        extensions = ["png", "jpg", "jpeg", "bmp"]

    filtered_paths = []
    for path in image_paths:
        extension = get_extension(path)
        if extension in extensions:
            filtered_paths.append(path)
    return filtered_paths


def get_extension(path):
    split_path = path.split(".")
    # Account for folders with names the same as supported file extensions
    if len(split_path) <= 1:
        return ""
    return split_path[-1]


if __name__ == "__main__":
    data_folder = "./data"
    image_folder = os.path.join(data_folder, "images")
    mask_folder = os.path.join(data_folder, "masks")

    main(
        image_folder,
        mask_folder,
    )