import os
from dataclasses import dataclass

import numpy as np

from climbing_thing.route import Route
from climbing_thing.climbnet.utils.visualizer import draw_instance_predictions
from climbing_thing.climbnet import ClimbNet, Instances
from climbing_thing.utils.image import imshow, mask
from climbing_thing.utils import logger
from copy import deepcopy
import cv2

log = logger.get_logger(__name__)
log.setLevel(logger.DEBUG_WITH_IMAGES)

@dataclass
class Point:
    x: int
    y: int
    scale: float

click_point = Point(x=-1, y=-1, scale=1)
old_click_point = Point(x=-1, y=-1, scale=1)
holds = None
selected_holds = set()


def main(image_dir, mask_dir):
    global holds, Point, selected_holds
    scale = 0.33
    click_point.scale = scale
    check_dirs(image_dir, mask_dir)

    image_paths = os.listdir(image_dir)
    image_paths = remove_non_image_paths(image_paths)

    climbnet_weights = "climbnet/weights/model_d2_R_50_FPN_3x.pth"
    model = ClimbNet(model_path=climbnet_weights, device="cuda")

    labeler_window_name = "labeler"
    cv2.namedWindow(labeler_window_name)
    cv2.setMouseCallback(labeler_window_name, click_event)

    saved_routes = []

    for image_path in image_paths:
        log.debug(f"Image: {image_path}")
        full_image_path = os.path.join(image_dir, image_path)
        route_image = cv2.imread(full_image_path)

        all_instances = model(route_image)
        all_instances = Instances(all_instances.instances.to("cpu"))
        holds = deepcopy(all_instances)

        old_holds = {-1}
        selected_holds = set()
        saved_routes = []

        while True:
            key = cv2.waitKey(1) & 0xFF

            if selected_holds != old_holds:
                instances = Instances(all_instances.instances[list(selected_holds)])
                image = draw_instance_predictions(
                    route_image, instances,
                    model.metadata
                )
                old_holds = set(selected_holds)

            if ord("q") == key:
                break
            elif ord("s") == key:
                log.info(f"Saved routes with holds: {selected_holds}")
                saved_routes.append(selected_holds)
                log.info(f"{len(saved_routes)} routes saved")
            elif ord("r") == key:
                log.info(f"Resetting selected holds")
                selected_holds = set()
            elif ord("w") == key:
                log.info(f"Saving {len(saved_routes)} routes")
                save_routes(all_instances, saved_routes, route_image)
                break

            imshow(labeler_window_name, image, scale=scale, delay=-1)


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global click_point, holds, selected_holds
        log.debug(f"click_event triggered at: {(x, y)}")
        click_point.x = x
        click_point.y = y
        x /= click_point.scale
        y /= click_point.scale
        x = int(x)
        y = int(y)

        for idx, mask in enumerate(holds.masks):
            mask = mask.to('cpu')
            mask = np.array(mask.long()).astype(np.uint8)[..., None]
            if mask[y, x] != 0:
                if idx in selected_holds:
                    selected_holds.remove(idx)
                else:
                    selected_holds.add(idx)
        log.debug(f"Selected holds: {selected_holds}")


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


def save_routes(all_instances, route_idxs, route_image):
    num_holds = len(all_instances)
    all_holds_idxs = set(range(num_holds))
    for hold_idxs in route_idxs:
        bad_idxs = all_holds_idxs - hold_idxs
        instances = Instances(all_instances.instances[list(bad_idxs)])
        output_mask = instances.combine_masks()
        output_mask = output_mask.max() - output_mask

        if log.level <= logger.DEBUG_WITH_IMAGES:
            viz_mask = (255 * output_mask).astype(np.uint8)
            viz_mask = cv2.cvtColor(viz_mask, cv2.COLOR_GRAY2BGR)
            masked_route = mask(route_image, output_mask)
            sbs = np.hstack([masked_route, viz_mask])
            imshow("Saved Route", sbs, scale=0.33, delay=0)
    cv2.destroyWindow("Saved Route")


if __name__ == "__main__":
    data_folder = "./data"
    image_folder = os.path.join(data_folder, "images")
    mask_folder = os.path.join(data_folder, "masks")

    main(
        image_folder,
        mask_folder,
    )