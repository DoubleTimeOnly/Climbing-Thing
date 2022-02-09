import os
from dataclasses import dataclass
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from .configs.model_params import model_params


class Instances:
    """Unwrap detectron2's Mask-RCNN output"""
    def __init__(self, model_output):
        self.instances = model_output
        self.boxes = self.instances.get("pred_boxes")
        self.scores = self.instances.get("scores")
        self.classes = self.instances.get("pred_classes")
        self.masks = self.instances.get("pred_masks")
        self.image_size = Size(
            height=self.instances.image_size[0],
            width=self.instances.image_size[1],
        )

    def combine_masks(self):
        """Combine each instance's binary mask into one image"""
        output_mask = np.zeros(self.image_size.as_tuple(), dtype=np.float32)
        for mask in self.masks:
            mask = mask.to('cpu')
            mask = np.array(mask.long()).astype(np.float32)
            output_mask += mask
        return output_mask

    def __len__(self):
        return len(self.instances)


class ClimbNet:
    def __init__(self, model_path, categories_file=None, device="cpu"):
        self.categories_file = categories_file
        if categories_file is None:
            self.categories_file = "climbnet/categories.json"
        dataset_name = "climb_dataset"
        register_coco_instances(dataset_name, {}, self.categories_file, "")
        self.config = self.setup_config(model_params)
        self.config.MODEL.WEIGHTS = os.path.join(model_path)
        self.config.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.config)
        self.metadata = MetadataCatalog.get(dataset_name)
        DatasetCatalog.get(dataset_name)

    def __call__(self, image: np.ndarray) -> Instances:
        """Runs detectron2 Mask-RCNN for instance segmentation"""
        outputs = self.predictor(image)
        return Instances(outputs["instances"])

    def setup_config(self, model_params: dict):
        config = get_cfg()
        config_file = model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        config.merge_from_file(config_file)
        config.DATALOADER.NUM_WORKERS = 1
        config.MODEL.ROI_HEADS.NUM_CLASSES = model_params["num_classes"]
        config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_params["score_test_threshold"]
        return config


@dataclass
class Size:
    width: int
    height: int

    def as_tuple(self):
        return self.height, self.width
