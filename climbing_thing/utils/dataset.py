import os

import torch

from climbing_thing import ROOT_DIR
from climbing_thing.metric_learning.dataset import ClassificationDataset


def compute_normalization_values(dataset: ClassificationDataset):
    means, stdevs = [], []
    for image, label in dataset:
        means.append(torch.mean(image, dim=[-1, -2]))
        stdevs.append(torch.std(image, dim=[-1, -2]))

    dataset_mean = torch.mean(torch.stack(means), dim=0)
    dataset_stdev = torch.mean(torch.stack(stdevs), dim=0)
    return dataset_mean, dataset_stdev


if __name__ == '__main__':
    images_path = os.path.join(ROOT_DIR, "data/instance_images/test2_masked")
    labels_csv = os.path.join(images_path, "test2_annotations.csv")
    dataset = ClassificationDataset(labels_csv, images_path)
    mean, stdev = compute_normalization_values(dataset)
    print(f"Dataset mean: {mean}")
    print(f"Dataset StDev: {stdev}")
    print(f"mean={mean.tolist()}, std={stdev.tolist()}")