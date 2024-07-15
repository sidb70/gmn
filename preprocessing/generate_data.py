import os
import sys
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.generate_nns import generate_random_cnn
from param_graph import seq_to_net



def train_random_cnns(
    n_architectures=10,
    train_size=None,
    batch_size=4,
    n_epochs=2,
    lr=0.001, 
    momentum=0.9,
    directory='data/cnn',
    **random_cnn_kwargs
):
    """
    Generates random CNN architectures and trains them on CIFAR10 data

    Args:
    - n_architectures: int, the number of architectures to generate
    - n_epochs: int, the number of epochs to train each architecture
    - other hyperparameters: see generate_random_cnn

    Saves these files to the specified directory:
    - features.pt: list of tuples (node_feats, edge_indices, edge_feats) for each model
    - accuracies.pt: list of accuracies for each model. 
    """

    torch.manual_seed(0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    if train_size is None:
        train_sampler = None
    else:
        train_sampler = SubsetRandomSampler(torch.randperm(len(trainset))[:train_size])
    
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

    testset = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    if train_size is None:
        test_sampler = None
    else:
        test_size = train_size // 4
        test_sampler = SubsetRandomSampler(torch.randperm(len(testset))[:test_size])
    
    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler)

    features = [[], [], []]
    accuracies = []

    for i in range(n_architectures):
        cnn = generate_random_cnn(**random_cnn_kwargs)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn.parameters(), lr=lr, momentum=momentum)

        for j in range(n_epochs):

            running_loss = 0.0
            for k, data in enumerate(trainloader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if k % 1 == 0:
                    print(f'\rModel {i+1}/{n_architectures}, Epoch {j+1}/{n_epochs}, Batch {k+1}/{len(trainloader)}, Loss: {running_loss/2000:.3f}', end='')
                    running_loss = 0.0

        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data

                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total

        print(f"\nAccuracy: {accuracy}\n")

        node_feats, edge_indices, edge_feats = seq_to_net(cnn).get_feature_tensors()

        features[0].append(node_feats)
        features[1].append(edge_indices)
        features[2].append(edge_feats)
        accuracies.append(accuracy)

    print('saving data')

    # ensure that the directory is valid and exists
    os.makedirs(directory, exist_ok=True)

    torch.save(features, os.path.join(directory, 'features.pt'))
    torch.save(accuracies, os.path.join(directory, 'accuracies.pt'))
