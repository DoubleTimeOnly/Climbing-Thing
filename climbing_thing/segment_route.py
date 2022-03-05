import cv2

from climbing_thing.climbnet import ClimbNet
from climbing_thing.climbnet.utils.visualizer import draw_instance_predictions
from climbing_thing.route.hue_difference import HueDifference
from climbing_thing.utils.image import imshow

route_rgb_colors = {
    "pink": (183, 77, 99),
    "orange": (190, 81, 63),
    "green": (57, 96, 64),
    "white": (170, 170, 170)
}

def segment_route():
    import argparse
    parser = argparse.ArgumentParser(description='Climbnet demo')
    parser.add_argument('--image_path', type=str,
                        required=False,
                        default="climbnet/test.png",
                        help='image file')
    args = parser.parse_args()

    # Read image
    test_image = cv2.imread(args.image_path)

    model = init_climbnet()
    hold_instances = model(test_image)

    route_segmentor = HueDifference(route_rgb_colors["white"], 15.0)
    route_instances = route_segmentor.segment_route(test_image, hold_instances)

    instance_image = draw_instance_predictions(test_image, route_instances, model.metadata)
    cv2.namedWindow("original image", cv2.WINDOW_NORMAL)
    cv2.imshow("original image", test_image)
    imshow("instances", instance_image)


def init_climbnet():
    default_weights = "climbnet/weights/model_d2_R_50_FPN_3x.pth"
    model = ClimbNet(model_path=default_weights, device="cpu")
    return model


if __name__ == '__main__':
    segment_route()
