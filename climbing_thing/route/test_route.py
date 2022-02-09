from unittest import TestCase
import cv2

from .route import Route
from climbing_thing.climbnet import ClimbNet
from climbing_thing.utils.image import imshow

class TestRoute(TestCase):
    def setUp(self):
        self.test_image = cv2.imread("climbnet/test.png")
        self.default_weights = "climbnet/weights/model_d2_R_50_FPN_3x.pth"

    def test_segment_pink_route(self):
        model = ClimbNet(model_path=self.default_weights, device="cuda")
        instances = model(self.test_image)

        pink_one_in_the_corner = Route(
            route_image=self.test_image,
            holds=instances,
            route_color=(183, 77, 99)
        )

        route_image = pink_one_in_the_corner.get_masked_route()
        assert len(pink_one_in_the_corner.holds) == 10
        # imshow("route_image", route_image, scale=0.3)
