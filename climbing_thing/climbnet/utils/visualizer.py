from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import Metadata
import numpy as np
from ..climbnet import Instances
from ...utils.image import imshow


def draw_instance_predictions(
        image: np.ndarray,
        model_output: Instances,
        train_metadata: Metadata,
        scale:float = 1.0
) -> np.ndarray:
    """Draw instances overlaid on input image"""
    viz = Visualizer(
        image, # [..., ::-1],
        metadata=train_metadata,
        scale=scale,
        instance_mode = ColorMode.SEGMENTATION
    )
    instance_viz = viz.draw_instance_predictions(model_output.instances.to('cpu'))
    return instance_viz.get_image()


def show_masks(model_output: Instances, scale=1.0):
    """Show predicted binary masks one instance at a time"""
    for mask in model_output.masks:
        mask = mask.to('cpu')
        mask = np.array(mask.long(), dtype=np.float32)
        imshow("masks", mask, scale=scale)

