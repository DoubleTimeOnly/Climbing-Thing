import csv
from enum import unique
import json
import os
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image



class ClassificationDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        with open(annotations_file, newline='') as annotations_fp:
            self.img_labels = [row for row in csv.reader(annotations_fp)][1:]
        self.class_map = {color:klass for (klass, color) in enumerate(set([color for (fp, color) in self.img_labels]))}
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = read_image(img_path).to(torch.float32) / 255
        label = torch.tensor(self.class_map[self.img_labels[idx][1]])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
