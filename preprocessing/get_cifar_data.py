from config import use_ffcv, cifar10_train_size
import os
import torch
import torchvision
from typing import List
if use_ffcv:
    print("Using ffcv")
    from .write_ffcv_data import cifar_10_to_beton
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Convert, Squeeze
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.fields.basics import Operation
else:
    print("Using torchvision")
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader

    from torchvision import transforms
    from torchvision.datasets import CIFAR10

def get_cifar_data(data_dir, device, batch_size, num_workers=1):
    if use_ffcv:
        if not os.path.exists(os.path.join(data_dir, "cifar_train.beton")):
            cifar_10_to_beton(data_dir)

        ###
        # https://docs.ffcv.io/ffcv_examples/cifar10.html
        CIFAR_MEAN = [125.307, 122.961, 113.8575]
        CIFAR_STD = [51.5865, 50.847, 51.255]
        loaders = {}
        for name in ["train", "test"]:
            label_pipeline: List[Operation] = [
                IntDecoder(),
                ToTensor(),
                ToDevice(device),
                Squeeze(),
            ]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            image_pipeline.extend(
                [
                    ToTensor(),
                    ToDevice(device, non_blocking=True),
                    ToTorchImage(),
                    Convert(torch.float32),
                    torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            )

            # Create loaders
            loaders[name] = Loader(
                os.path.join(data_dir, f"cifar_{name}.beton"),
                batch_size=batch_size,
                num_workers=num_workers,
                order=OrderOption.RANDOM,
                drop_last=(name == "train"),
                pipelines={"image": image_pipeline, "label": label_pipeline},
            )
        ###
        trainloader = loaders["train"]
        testloader = loaders["test"]

    else:
        cifar_10_dir = os.path.join(data_dir, "cifar10")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = CIFAR10(
            root=cifar_10_dir, train=True, download=True, transform=transform
        )
        train_sampler = SubsetRandomSampler(
            torch.randperm(len(trainset))[:cifar10_train_size]
        )

        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

        testset = CIFAR10(
            root=cifar_10_dir, train=False, download=True, transform=transform
        )
        test_size = cifar10_train_size // 4
        test_sampler = SubsetRandomSampler(torch.randperm(len(testset))[:test_size])

        testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler)
    # trainloader.to(device)
    # testloader.to(device)
    return trainloader, testloader