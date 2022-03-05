import sys

import cv2
import numpy as np
from climbing_thing.climbnet import ClimbNet
from climbing_thing.climbnet.utils.visualizer import draw_instance_predictions, show_masks
from climbing_thing.utils.image import imshow


def show_instances_and_binary_masks(input_image, output):
    combined_mask = output.combine_masks()
    image = draw_instance_predictions(input_image, output, model.metadata, scale=1.0)

    combined_mask = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    combined_mask = (255 * combined_mask).astype(np.uint8)
    side_by_side = np.hstack([image, combined_mask])
    imshow("instances", side_by_side, scale=1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Climbnet demo')
    parser.add_argument('--image_path', type=str,
                        required=False,
                        default="climbnet/test.png",
                        help='image file')
    args = parser.parse_args()

    # Read image
    test_image = cv2.imread(args.image_path)

    default_weights = "climbnet/weights/model_d2_R_50_FPN_3x.pth"
    model = ClimbNet(model_path=default_weights, device="cpu")
    output = model(test_image)

    show_instances_and_binary_masks(test_image, output)

