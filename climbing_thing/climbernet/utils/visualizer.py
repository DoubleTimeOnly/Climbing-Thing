from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import Metadata
import numpy as np
from ..climbernet import KeyPointInstances
from ...utils.image import imshow


def draw_instance_predictions(
        image: np.ndarray,
        model_output: KeyPointInstances,
        train_metadata: Metadata,
        scale:float = 1.0
) -> np.ndarray:
    """Draw instances overlaid on input image"""
    viz = Visualizer(
        image, # [..., ::-1],
        metadata=train_metadata,
        scale=scale,
        instance_mode = ColorMode.SEGMENTATION,
    )
    instance_viz = viz.draw_and_connect_keypoints(model_output.keypoints.to('cpu')[0])
    return instance_viz.get_image()


def show_masks(model_output: KeyPointInstances, scale=1.0):
    """Show predicted binary masks one instance at a time"""
    for mask in model_output.masks:
        mask = mask.to('cpu')
        mask = np.array(mask.long(), dtype=np.float32)
        imshow("masks", mask, scale=scale)

