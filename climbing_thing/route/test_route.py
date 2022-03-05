from unittest import TestCase
import cv2

from .hue_difference import HueDifference
from climbing_thing.climbnet import ClimbNet
from climbing_thing.utils.image import imshow

class TestHueDifference(TestCase):
    def setUp(self):
        self.test_image = cv2.imread("climbnet/test.png")
        self.default_weights = "climbnet/weights/model_d2_R_50_FPN_3x.pth"

    def test_segment_pink_route(self):
        hold_model = ClimbNet(model_path=self.default_weights, device="cuda")
        hold_instances = hold_model(self.test_image)

        route_segmentor = HueDifference(
            route_color=(183, 77, 99),
            hue_difference_threshold=10
        )
        
        route_instances = route_segmentor.segment_route(self.test_image, hold_instances)

        assert len(route_instances) == 10
        # imshow("route_image", route_image, scale=0.3)
