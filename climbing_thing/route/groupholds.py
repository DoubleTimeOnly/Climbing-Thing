from typing import List, Dict

import cv2
import torch

from climbing_thing.climbnet import Instances
from climbing_thing.utils.mennard import colors_probs
import numpy as np



COLORS = [
    "purple",
    "white",
    "green",
    "yellow",
    "black",
    "orange",
    "blue"
]


def segment_routes(wall_image: np.ndarray, holds: Instances) -> Dict[str, List[int]]:
    # run mennard -> each hold image
    # get the class (color) of each hold

    # wall_image = # torch.as_tensor(wall_image.astype("float32").transpose(2, 0, 1))

    # iterative over all masks and get hold images
    batch: List[torch.Tensor] = []
    for idx, mask in enumerate(holds.masks):
        mask = mask.to('cpu')   # same size as wall image
        mask = np.array(mask.long()).astype(np.uint8)
        hold_bbox = holds.boxes[idx].tensor.int()   # 20x40 pixel magnitude

        # masked_image = wall_image[mask > 0] # same size as wall_image
        masked_image = cv2.bitwise_and(wall_image, wall_image, mask=mask)
        # hold_image = wall_image[hold_bbox[0, 1]:hold_bbox[0, 3], hold_bbox[0, 0]:hold_bbox[0, 2]]
        hold_image = masked_image[hold_bbox[0, 1]:hold_bbox[0, 3], hold_bbox[0, 0]:hold_bbox[0, 2]]

        # debugging
        # cv2.imshow("image", hold_image)
        # cv2.waitKey()
        hold_image = hold_image[..., ::-1].copy()
        batch.append(hold_image)

    prompts = COLORS


    probs = colors_probs(batch, prompts)    # NxP

    route_dict = {c: [] for c in prompts}
    for hold_idx, prob in enumerate(probs):
        color_idx = np.argmax(prob)
        route_dict[prompts[color_idx]].append(hold_idx)

    return route_dict


def viz_tensor(image: torch.Tensor):
    image = image.clone().transpose(1, 2, 0).numpy()
    cv2.imshow("image", image)
    cv2.waitKey()
