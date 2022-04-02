from unittest import TestCase
import cv2

from .climbernet import ClimberNet
from climbing_thing.climbernet.utils import visualizer as viz
from climbing_thing.utils.image import imshow


class TestClimberNet(TestCase):
    def setUp(self):
        self.test_image = cv2.imread("climbnet/test2.png")
        self.default_weights = "climbernet/weights/model_final_5ad38f.pkl"

    def test_forward_pass(self):
        model = ClimberNet(model_path=self.default_weights, device="cuda")
        instances = model(self.test_image)

        output_image = viz.draw_instance_predictions(self.test_image, instances, model.predictor.metadata, scale=0.3)
        imshow("keypoints", output_image, delay=0)
