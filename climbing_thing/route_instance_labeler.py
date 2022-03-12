import os
from dataclasses import dataclass

import numpy as np

from climbing_thing.climbnet.utils.visualizer import draw_instance_predictions
from climbing_thing.climbnet import ClimbNet, Instances
import climbing_thing.utils.image as imutils
from climbing_thing.utils.image import imshow, mask, hstack
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


class RouteLabeler:
    def __init__(self, image_dir, mask_dir, show_scale=0.33):
        check_dirs(image_dir, mask_dir)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        image_paths = os.listdir(image_dir)
        image_paths = remove_non_image_paths(image_paths)
        self.image_paths = image_paths

        climbnet_weights = "climbnet/weights/model_d2_R_50_FPN_3x.pth"
        self.model = ClimbNet(model_path=climbnet_weights, device="cuda")

        self.labeler_window_name = "labeler"
        cv2.namedWindow(self.labeler_window_name)
        cv2.setMouseCallback(self.labeler_window_name, self.click_event)

        self.holds = None
        self.selected_holds = set()
        self.mask_index = None
        self.scale = show_scale

    def label_images(self):
        for image_path in self.image_paths:
            log.debug(f"Image: {image_path}")
            full_image_path = os.path.join(self.image_dir, image_path)
            route_image = cv2.imread(full_image_path)

            self.holds, self.mask_index = self.get_holds(route_image)

            old_holds = {-1}
            self.selected_holds = set()
            saved_routes = []

            while True:
                key = cv2.waitKey(1) & 0xFF

                if self.selected_holds != old_holds:
                    instances = Instances(self.holds.instances[list(self.selected_holds)])
                    image = draw_instance_predictions(
                        route_image, instances,
                        self.model.metadata
                    )
                    old_holds = set(self.selected_holds)

                if ord("q") == key:
                    break
                elif ord("s") == key:
                    log.info(f"Saved routes with holds: {self.selected_holds}")
                    saved_routes.append(self.selected_holds)
                    log.info(f"{len(saved_routes)} routes saved")
                elif ord("r") == key:
                    log.info(f"Resetting selected holds")
                    self.selected_holds = set()
                elif ord("w") == key:
                    log.info(f"Saving {len(saved_routes)} routes")
                    self.save_routes(
                        saved_routes,
                        route_image,
                        image_path,
                    )
                    break

                imshow(self.labeler_window_name, image, scale=self.scale, delay=-1)

    def get_holds(self, route_image):
        all_instances = self.model(route_image)
        all_instances = Instances(all_instances.instances.to("cpu"))
        holds = deepcopy(all_instances)
        mask_index = self.index_holds(holds)
        return holds, mask_index

    @staticmethod
    def index_holds(holds: Instances) -> np.ndarray:
        """
        Return a mask whose pixel location stores the index of the hold instance it belongs to
        -1 means no instance
        0 means 0th hold instance
        output shape: (mask_height, mask_width)
        """
        mask_index = None
        for idx, mask in enumerate(holds.masks):
            # mask = mask.to('cpu')
            mask = np.array(mask, dtype=np.int32)     # range: [0, 1]

            if idx == 0:
                mask_index = mask - 1
            else:
                mask_index[mask == 1] = idx
        return mask_index


    def save_routes(self, route_idxs, route_image, image_path):
        for i, hold_idxs in enumerate(route_idxs):
            # TODO: save composite mask with both on (green) and off (red) holds
            inverted_mask = self.hold_idxs_to_mask(hold_idxs)

            viz_mask = imutils.float_to_int(inverted_mask)
            self.save_mask(image_path, i, viz_mask)

            if log.level <= logger.DEBUG_WITH_IMAGES:
                masked_route = mask(route_image, inverted_mask)
                sbs = imutils.hstack([masked_route, viz_mask])
                imshow("Saved Route", sbs, scale=self.scale, delay=0)

        if log.level <= logger.DEBUG_WITH_IMAGES:
            cv2.destroyWindow("Saved Route")

    def hold_idxs_to_mask(self, hold_idxs):
        all_holds_idxs = set(range(len(self.holds)))
        bad_idxs = all_holds_idxs - hold_idxs

        instances = Instances(self.holds.instances[list(bad_idxs)])
        output_mask = instances.combine_masks()

        inverted_mask = output_mask.max() - output_mask
        return inverted_mask

    def save_mask(self, image_path, mask_idx, mask):
        basename = image_path.split(".")[-2]
        filename = f"{basename}_mask_{mask_idx}.png"
        mask_path = os.path.join(self.mask_dir, filename)
        cv2.imwrite(mask_path, mask)
        log.debug(f"Writing: {os.path.abspath(mask_path)}")

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            log.debug(f"click_event triggered at: {(x, y)}")
            x, y = int(x / self.scale), int(y / self.scale)
            idx = self.is_hold(x, y)

            if idx != -1:
                if idx in self.selected_holds:
                    self.selected_holds.remove(idx)
                else:
                    self.selected_holds.add(idx)
            log.debug(f"Selected holds: {self.selected_holds}")

    def is_hold(self, x, y):
        hold_idx = self.mask_index[y, x]
        return hold_idx


def main(image_dir, mask_dir):
    labeler = RouteLabeler(image_dir, mask_dir, show_scale=0.25)
    labeler.label_images()


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