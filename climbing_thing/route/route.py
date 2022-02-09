from climbing_thing.climbnet import Instances
import climbing_thing.utils.image as imutils
import numpy as np
import cv2


class Route:
    def __init__(
            self,
            route_image: np.ndarray,
            holds: Instances,
            route_color: tuple,
            hue_tolerance: int = 10,
    ):
        """
        :param route_color: RGB
        """
        self.route_image = route_image
        self.holds = holds  # All holds for now
        rgb_color = np.array([[route_color]], dtype=np.uint8)[..., ::-1]
        hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2HSV)
        self.route_color = hsv_color[0, 0]
        self.hue_tolerance = hue_tolerance
        hold_idxs = self.segment_route()
        self.holds = Instances(holds.instances[hold_idxs])

    def segment_route(self):
        """Return indexes of holds that are of specified route color"""
        # TODO: experiment between simple color range vs. dbscan on hue histogram
        route_hold_idxs = []
        hsv_image = cv2.cvtColor(self.route_image, cv2.COLOR_BGR2HSV)
        for idx, mask in enumerate(self.holds.masks):
            mask = mask.to('cpu')
            mask = np.array(mask.long()).astype(np.uint8)[..., None]
            # hold_image = imutils.mask(hsv_image, mask)

            if self.hold_is_on_route(hsv_image, mask):
                route_hold_idxs.append(idx)

        if len(route_hold_idxs) == 0:
            raise Exception("Could not find any holds with target color")

        return route_hold_idxs

    def hold_is_on_route(self, hsv_image, hold_mask):
        # TODO: take into account hue wrapping at 180
        avg_hue = cv2.mean(hsv_image, mask=hold_mask)[0]
        target_hue = self.route_color[0]
        # print(avg_hue, target_hue)
        hue_distance = abs(avg_hue - target_hue)
        hue_distance = min(hue_distance, 180 - hue_distance)

        if hue_distance < self.hue_tolerance:
            return True
        return False

    def get_masked_route(self):
        binary_mask = self.holds.combine_masks()
        masked_route = imutils.mask(self.route_image, binary_mask)
        return masked_route
