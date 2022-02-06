from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import Metadata
import cv2
import numpy as np
from ..climbnet import Instances


def draw_instance_predictions(
        image: np.ndarray,
        model_output: Instances,
        train_metadata: Metadata,
        scale:float = 1.0
) -> np.ndarray:
    """Draw instances overlaid on input image"""
    viz = Visualizer(
        image[..., ::-1],
        metadata=train_metadata,
        scale=scale,
        instance_mode=ColorMode.IMAGE_BW
    )
    instance_viz = viz.draw_instance_predictions(model_output.instances.to('cpu'))
    return instance_viz.get_image()


def show_masks(model_output: Instances, scale=1.0):
    """Show predicted binary masks one instance at a time"""
    for mask in model_output.masks:
        mask = mask.to('cpu')
        mask = np.array(mask.long(), dtype=np.float32)
        imshow("masks", mask, scale=scale)


def imshow(window_name: str, image: np.ndarray, scale: float = 1.0, delay=0):
    output = image.copy()
    if scale != 1.0:
        output = cv2.resize(output, dsize=(0, 0), fx=scale, fy=scale)
    cv2.imshow(window_name, output)
    if delay >= 0:
        cv2.waitKey(delay)
