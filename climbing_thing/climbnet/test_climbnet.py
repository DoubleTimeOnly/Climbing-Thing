import cv2
from .climbnet import ClimbNet

from unittest import TestCase



class TestClimbNet(TestCase):
    def setUp(self):
        self.test_image = cv2.imread("climbnet/test.png")
        self.default_weights = "climbnet/weights/model_d2_R_50_FPN_3x.pth"

    def test_forward_pass(self):
        model = ClimbNet(model_path=self.default_weights, device="cuda")
        mask = model(self.test_image)

