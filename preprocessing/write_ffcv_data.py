"""
https://docs.ffcv.io/ffcv_examples/cifar10.html
"""

from argparse import ArgumentParser
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import torchvision
import os


def cifar_10_to_beton(save_path):
    datasets = {
        "train": torchvision.datasets.CIFAR10(save_path, train=True, download=True),
        "test": torchvision.datasets.CIFAR10(save_path, train=False, download=True),
    }

    for name, ds in datasets.items():
        writer = DatasetWriter(
            os.path.join(save_path, f"cifar_{name}.beton"),
            {"image": RGBImageField(), "label": IntField()},
        )
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./data/cifar10")
    args = parser.parse_args()
    save_path = args.save_path
    cifar_10_to_beton(save_path)
