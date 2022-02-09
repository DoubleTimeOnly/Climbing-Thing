import sys

import cv2

from climbing_thing.route import Route
from climbing_thing.climbnet import ClimbNet
from climbing_thing.climbnet.utils.visualizer import draw_instance_predictions
from climbing_thing.utils.image import imshow

route_rgb_colors = {
    "pink": (183, 77, 99),
    "orange": (190, 81, 63),
    "green": (57, 96, 64)
}

def segment_route():
    import argparse
    parser = argparse.ArgumentParser(description='Climbnet demo')
    parser.add_argument('--image_path', type=str,
                        required=False,
                        help='image file')
    args = parser.parse_args()

    # Read image
    if len(sys.argv) <= 1:
        image_path = "climbnet/test.png"
    else:
        image_path = args.image_path
    test_image = cv2.imread(image_path)

    model = init_climbnet()
    output = model(test_image)
    route = Route(
        route_image=test_image,
        holds=output,
        route_color=route_rgb_colors["pink"]
    )
    imshow("route image", route.get_masked_route(), scale=0.3, delay=-1)
    instance_image = draw_instance_predictions(test_image, route.holds, model.metadata)
    imshow("instances", instance_image, scale=0.3)


def init_climbnet():
    default_weights = "climbnet/weights/model_d2_R_50_FPN_3x.pth"
    model = ClimbNet(model_path=default_weights, device="cpu")
    return model


if __name__ == '__main__':
    segment_route()
