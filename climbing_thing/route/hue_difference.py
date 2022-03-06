from climbing_thing.climbnet import Instances
import climbing_thing.utils.image as imutils
import numpy as np
import cv2


class HueDifference:
    def __init__(self, route_color: tuple, hue_difference_threshold: float) -> None:
        """
        :param target_color: RGB
        :param color_threshold: degrees
        """
        self.route_color_rgb = route_color
        self.hue_difference_threshold = hue_difference_threshold
        bgr_color = np.array([[route_color]], dtype=np.uint8)[..., ::-1]
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
        self.route_color_hsv = hsv_color[0, 0]

    def segment_route(self, image, holds) -> Instances:
        """Return Instances of holds that are of specified route color"""
        # TODO: experiment between simple color range vs. dbscan on hue histogram
        route_hold_idxs = []
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for idx, mask in enumerate(holds.masks):
            mask = mask.to('cpu')
            mask = np.array(mask.long()).astype(np.uint8)[..., None]
            # hold_image = imutils.mask(hsv_image, mask)

            if self.hold_is_on_route(hsv_image, mask):
                route_hold_idxs.append(idx)

        if len(route_hold_idxs) == 0:
            raise Exception("Could not find any holds with target color")

        return Instances(holds.instances[route_hold_idxs])

    def hold_is_on_route(self, hsv_image, hold_mask):
        hues = hsv_image[..., 0]
        masked_hues = hues[hold_mask[..., 0] > 0]
        rad_mean = self.circular_mean(np.deg2rad(masked_hues))
        avg_hue = np.rad2deg(rad_mean)
        target_hue = self.route_color_hsv[0]
        hue_distance = abs(avg_hue - target_hue)
        hue_distance = min(hue_distance, 180 - hue_distance)

        return hue_distance < self.hue_difference_threshold
           

    @staticmethod
    def circular_mean(angles):
        return np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())

    def get_masked_route(self):
        binary_mask = self.holds.combine_masks()
        masked_route = imutils.mask(self.route_image, binary_mask)
        return masked_route
