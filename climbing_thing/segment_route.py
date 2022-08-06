import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from climbing_thing import ROOT_DIR
from climbing_thing.climbnet import ClimbNet, Instances
from climbing_thing.climbnet.utils.visualizer import draw_instance_predictions
from climbing_thing.metric_learning.models import Net
from climbing_thing.route.compareholds import metric_distances
from climbing_thing.route.histogram_clustering import HistogramClustering
from climbing_thing.route.hue_difference import HueDifference
from climbing_thing.utils.distancemetrics import compute_histograms
from climbing_thing.utils.image import imshow, crop_image, mask_image
from climbing_thing.utils.performancemetrics import PerformanceMetrics
import climbing_thing.utils.image as imutils


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
    default_weights = os.path.join(ROOT_DIR, "climbnet/weights/model_d2_R_50_FPN_3x.pth")
    model = ClimbNet(model_path=default_weights, device="cpu")
    return model


def get_route_of_same_color_or_index():
    from climbing_thing.route.groupholds import segment_routes
    # image_path = "climbnet/test2.png"
    image_path = "./data/images/20220207_215203.jpg"
    image_file = os.path.join(ROOT_DIR, image_path)
    wall_image = cv2.imread(image_file)

    climbnet = init_climbnet()

    holds = climbnet(wall_image)

    routes = segment_routes(wall_image, holds)

    routes = {c: Instances(holds.instances[hold_idxs]) for c, hold_idxs in routes.items()}

    for color, route in routes.items():
        binary_mask = route.combine_masks()
        masked_route = imutils.mask(wall_image, binary_mask)
        imutils.imshow("route", masked_route)

    # run mennard -> each hold image
    # get the class (color) of each hold
    # combine the masks of same class/color
    # apply mask to wall image to get segmented route image
    # just the holds, holds + wall = wall image - irrelevant holds (latter is good)
    # output -> route_classifier_model() -> grade
    # holds + wall, wall image + mask (of just holds)
    #                   ^ is the ideal scenario
    #                       how do you integrate the mask into the model?
    # wall image -> a few layers of network -> you get some input/4xinput/4xdepth feature map
    # you literally mask out some values -> resize the mask as well
    # why we might want to consider just the holds
        # you can take advantage of current architectures
        # you might not have to train as many images
    # example: take CLIP's RN50 classifier and use those weights as a base
    # finetune/train RN50 on our classification images

    # Today
    # build the thing where we feed
    # we get routes from hold instances
        # labeling
        # eventual final goal

    # final goal?
    # choose some route through some method -> returns the grade of that route


def save_instances_with_idx():
    image_file = "climbnet/test2.png"
    test_image = cv2.imread(image_file)
    model = init_climbnet()
    holds = model(test_image)

    image_folder = os.path.join(ROOT_DIR, "data/instance_images/test2_masked")
    os.makedirs(image_folder, exist_ok=True)
    cv2.imwrite(f"{image_folder}/test2.png", test_image)

    for hold_idx, mask in enumerate(holds.masks):
        bbox = holds.boxes[hold_idx].tensor[0]
        mask = holds.masks[hold_idx]
        hold_image = crop_image(test_image, bbox)
        mask = crop_image(mask, bbox)
        masked_hold_image = mask_image(hold_image, mask)
        cv2.imwrite(f"{image_folder}/hold_{hold_idx}.png", masked_hold_image)


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
        (h_hist, h_edges), (s_hist, s_edges), (v_hist, v_edges) = compute_histograms(hsv_image, bins=180, mask=mask, mode="np")


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
    # save_instances_with_idx()
    # save_histogram_instances_with_idx()
    get_route_of_same_color_or_index()