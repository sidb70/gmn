from gmn_lim.model_arch_graph import seq_to_feats
import os
import sys
import torch
import torch.nn as nn
from torch import optim
import torchvision
import numpy as np
from dataclasses import dataclass

use_ffcv = True
if use_ffcv:
    from .write_ffcv_data import cifar_10_to_beton
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice,  RandomHorizontalFlip, RandomTranslate, Cutout, ToTorchImage, Convert, Squeeze
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.fields.basics import Operation
else:
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader

    from torchvision import transforms
    from torchvision.datasets import CIFAR10

from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.generate_nns import generate_random_cnn, RandCNNConfig

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@dataclass
class Hyperparameters:
    """
    Dataclass for hyperparameters to use for training CNNs.
    """
    batch_size: int = 3
    lr: float = 0.01
    n_epochs: int = 1
    momentum: float = 0.5

    def to_vec(self):
        return [self.batch_size, self.lr, self.n_epochs, self.momentum]
    
    def __str__(self):
        return f"batch_size: {self.batch_size}, lr: {self.lr}, n_epochs: {self.n_epochs}, momentum: {self.momentum}"


def train_random_cnns_hyperparams(
    n_architectures=10,
    directory='data/cnn_hpo',
    replace_if_existing=True, 
):
    """
    Generates and trains random CNNs, using random hyperparameters
    """

    features = []
    accuracies = []

    for i in range(n_architectures):
        batch_size = np.random.randint(2, 1024)
        lr = np.random.uniform(0.0001, 0.1)
        n_epochs = np.random.randint(50, 150)
        momentum = np.random.uniform(0.1, 0.9)
        hyperparams = Hyperparameters(batch_size, lr, n_epochs, momentum)

        print("training random cnn with hyperparameters:")
        print(f"batch_size: {batch_size}")
        print(f"lr: {lr}")
        print(f"n_epochs: {n_epochs}")
        print(f"momentum: {momentum}")

        feats, acc = train_random_cnns(
            n_architectures=1,
            hyperparams=hyperparams,
            directory=directory,
            replace_if_existing=replace_if_existing,
            save=True,
        )
        features.append(feats[0])
        accuracies.append(acc[0])



def train_random_cnns(
    hyperparams=Hyperparameters(),
    random_cnn_config=RandCNNConfig(),
    n_architectures=10,
    num_workers=4,
    optimizer_type=None,
    data_dir='data',
    results_dir='data/cnn',
    save=True,
    replace_if_existing=False,
):
    """
    Generates random CNN architectures and trains them on CIFAR10 data
    Saves the resulting CNN node and edge features and their accuracies in directory

    Args:
    - hyperparams: Hyperparameters, the hyperparameters to use for training
    - random_cnn_config: RandomCNNConfig, the configuration for generating random CNNs
    - n_architectures: int, the number of architectures to generate
    - optimizer_type: torch.optim.Optimizer, the optimizer to use. If None, uses SGD
    - data_dir: str, the directory in which the CIFAR10 data is stored
    - results_dir: str, the directory in which to save the CNN features and their accuracies
    - save: bool, whether to save the results to results_dir
    - replace_if_existing: bool, whether to replace the existing results if they exist or append to them

    Saves these files to the specified directory:
    - {results_dir}/features.pt: list of tuples (node_feats, edge_indices, edge_feats) for each model
    - {results_dir}/accuracies.pt: list of accuracies for each model.
    """

    hpo_vec = hyperparams.to_vec()
    batch_size, lr, n_epochs, momentum = hpo_vec

    cifar_10_dir = os.path.join(data_dir, 'cifar10')

    if use_ffcv:
        if not os.path.exists(os.path.join(cifar_10_dir, 'cifar_train.beton')):
            cifar_10_to_beton(cifar_10_dir)

        ###
        #https://docs.ffcv.io/ffcv_examples/cifar10.html
        CIFAR_MEAN = [125.307, 122.961, 113.8575]
        CIFAR_STD = [51.5865, 50.847, 51.255]
        loaders = {}
        for name in ['train', 'test']:
            label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            image_pipeline.extend([
                ToTensor(),
                ToDevice(DEVICE, non_blocking=True),
                ToTorchImage(),
                Convert(torch.float32),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])

            # Create loaders
            loaders[name] = Loader(os.path.join(cifar_10_dir,'cifar10', f'cifar_{name}.beton'),
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    order=OrderOption.RANDOM,
                                    drop_last=(name == 'train'),
                                    pipelines={'image': image_pipeline,
                                            'label': label_pipeline})
        ###                             
        trainloader = loaders['train']  
        testloader = loaders['test']    

    else:
        train_size=1000
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        train_sampler = SubsetRandomSampler(torch.randperm(len(trainset))[:train_size])

        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

        testset = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
        test_size = train_size // 4
        test_sampler = SubsetRandomSampler(torch.randperm(len(testset))[:test_size])

        testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler)


    features = []
    accuracies = []

    for i in range(n_architectures):
        cnn = generate_random_cnn(random_cnn_config).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        if optimizer_type is None:
            optimizer = optim.SGD(cnn.parameters(), lr=lr, momentum=momentum)
        else:
            optimizer = optimizer_type(cnn.parameters(), lr=lr, momentum=momentum)

        for j in range(n_epochs):

            running_loss = 0.0
            for k, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                outputs = cnn(inputs).reshape(-1,10)
                labels = labels.reshape(-1)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                if k % 20 == 0:
                    print(f'\rTraining model {i+1}/{n_architectures}, Epoch {j+1}/{n_epochs}, Batch {k+1}/{len(trainloader)}, Loss: {running_loss:.3f}', end='')
                    running_loss = 0.0

        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = cnn(images).reshape(-1,10)
                labels = labels.reshape(-1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total

        print(f"\nAccuracy: {accuracy}")
        node_feats, edge_indices, edge_feats = seq_to_feats(cnn)
        features.append((node_feats, edge_indices, edge_feats, hpo_vec))
        accuracies.append(accuracy)


    if save:
        os.makedirs(results_dir, exist_ok=True)

        if replace_if_existing:
            # if feats and accuracies already exist, delete them and save the new data.
            features_path = os.path.join(results_dir, 'features.pt')
            accuracies_path = os.path.join(results_dir, 'accuracies.pt')
            if os.path.exists(features_path):
                os.remove(features_path)
            if os.path.exists(accuracies_path):
                os.remove(accuracies_path)

        else:
            # if feats and accuracies both exist, then append them to the current features and accuracies
            if os.path.exists(os.path.join(results_dir, 'features.pt')) and os.path.exists(os.path.join(results_dir, 'accuracies.pt')):
                features = torch.load(os.path.join(results_dir, 'features.pt')) + features
                accuracies = torch.load(os.path.join(results_dir, 'accuracies.pt')) + accuracies

        torch.save(features, os.path.join(results_dir, 'features.pt'))
        torch.save(accuracies, os.path.join(results_dir, 'accuracies.pt'))

    return features, accuracies
