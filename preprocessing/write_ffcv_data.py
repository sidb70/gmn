"""
https://docs.ffcv.io/ffcv_examples/cifar10.html
"""

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import torchvision
from torch.utils.data import random_split
import os


def cifar_10_to_beton(save_path):
    # Load the CIFAR-10 dataset
    full_train_dataset = torchvision.datasets.CIFAR10(save_path, train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(save_path, train=False, download=True)

    # Split the training dataset into training and validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    datasets = {
        "train": train_dataset,
        "valid": val_dataset,
        "test": test_dataset,
    }

    for name, ds in datasets.items():
        writer = DatasetWriter(
            os.path.join(save_path, f"cifar_{name}.beton"),
            {"image": RGBImageField(), "label": IntField()},
        )
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    save_path = "./data/"
    cifar_10_to_beton(save_path)
