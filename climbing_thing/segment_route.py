import cv2
import numpy as np
import matplotlib.pyplot as plt

from climbing_thing.climbnet import ClimbNet
from climbing_thing.climbnet.utils.visualizer import draw_instance_predictions
from climbing_thing.route.compareholds import compute_cartesian_difference
from climbing_thing.route.histogram_clustering import HistogramClustering
from climbing_thing.route.hue_difference import HueDifference
from climbing_thing.utils.distancemetrics import compute_hsv_histogram
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

    # this is the orange start hold in test2.png
    # target_hold = hold_instances.instances[77]

    route_segmentor = HistogramClustering(target_hold)
    # for i in range(7):        
    #     route_instances = route_segmentor.segment_route(test_image, hold_instances, i)
    #     instance_image = draw_instance_predictions(test_image, route_instances, model.metadata)
    #     cv2.namedWindow("original image", cv2.WINDOW_NORMAL)
    #     cv2.imshow("original image", test_image)
    #     imshow("instances", instance_image)

    for color_name, color in route_rgb_colors.items():
        route_segmentor = HueDifference(color, 15)
        route_instances = route_segmentor.segment_route(test_image, hold_instances)
        instance_image = draw_instance_predictions(test_image, route_instances, model.metadata)
        cv2.namedWindow("original image", cv2.WINDOW_NORMAL)
        cv2.imshow("original image", test_image)
        imshow(f"instances ({color_name})", instance_image)


def init_climbnet():
    default_weights = "climbnet/weights/model_d2_R_50_FPN_3x.pth"
    model = ClimbNet(model_path=default_weights, device="cpu")
    return model


def compare_holds():
    image_file = "climbnet/test2.png"
    test_image = cv2.imread(image_file)
    model = init_climbnet()
    hold_instances = model(test_image)

    compute_cartesian_difference(test_image, hold_instances)


def save_instances_with_idx():
    image_file = "climbnet/test2.png"
    test_image = cv2.imread(image_file)
    model = init_climbnet()
    holds = model(test_image)

    image_folder = "data/instance_images/test2"
    cv2.imwrite(f"{image_folder}/test2.png", test_image)

    for idx, mask in enumerate(holds.masks):
        hold_bbox = holds.boxes[idx].tensor.int()
        hold_image = test_image[hold_bbox[0, 1]:hold_bbox[0, 3], hold_bbox[0, 0]:hold_bbox[0, 2]]
        cv2.imwrite(f"{image_folder}/hold_{idx}.png", hold_image)


def save_histogram_instances_with_idx():
    image_file = "climbnet/test2.png"
    test_image = cv2.imread(image_file)
    model = init_climbnet()
    holds = model(test_image)

    histogram_folder = "data/instance_histograms/test2"
    cv2.imwrite(f"{histogram_folder}/test2.png", test_image)

    hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    for idx, mask in enumerate(holds.masks):
        mask = mask.to("cpu")
        mask = np.array(mask.long()).astype(np.uint8)
        (h_hist, h_edges), (s_hist, s_edges), (v_hist, v_edges) = compute_hsv_histogram(hsv_image, bins=180, mask=mask, mode="np")


        fig, ax = plt.subplots(1, 3)
        ax[0].stairs(h_hist, h_edges, color='b')
        ax[0].set_ylim(0, 0.4)

        ax[1].stairs(s_hist, s_edges, color='g')
        ax[1].set_ylim(0, 0.4)

        ax[2].stairs(v_hist, v_edges, color='r')
        ax[2].set_ylim(0, 0.4)

        plt.tight_layout()
        plt.savefig(f"{histogram_folder}/hold_{idx}.png")


if __name__ == '__main__':
    # segment_route()
    # compare_holds()
    # save_instances_with_idx()
    save_histogram_instances_with_idx()
