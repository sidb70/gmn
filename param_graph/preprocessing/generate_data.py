import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import torch
from param_graph.preprocessing.generate_nns import generate_random_cnn
from param_graph.gmn_lim.model_arch_graph import sequential_to_feats

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def generate_data(
    n_architectures: int,
    train_proportion: int,
    n_epochs: int,
    batch_size=32,
    lr=0.001, 
    momentum=0.9,
    directory='data/cnn_data'
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_size = int(train_proportion * len(trainset))
    train_sampler = SubsetRandomSampler(torch.randperm(len(trainset))[:train_size])

    
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

    testset = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    if train_size is None:
        test_sampler = None
    else:
        test_size = train_size // 4
        test_sampler = SubsetRandomSampler(torch.randperm(len(testset))[:test_size])
    
    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler)

    features = []
    accuracies = []
    print("Num train samples", len(trainloader) * batch_size)
    for i in range(n_architectures):
        cnn = generate_random_cnn().to(DEVICE)
        print("Created CNN:", cnn)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn.parameters(), lr=lr, momentum=momentum)

        for j in range(n_epochs):
            running_loss = 0.0
            for k, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if k % 100 == 99:
                    print(f'\rModel {i+1}/{n_architectures}, Epoch {j+1}/{n_epochs}, Batch {k+1}/{len(trainloader)}, Loss: {running_loss/2000:.3f}', end='')
                    running_loss = 0.0

        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total

        print(f"\nAccuracy: {accuracy}")

        print("Creating parameter graph with ", sum(p.numel() for p in cnn.parameters()), " parameters")
        # node_feats, edge_indices, edge_feats = seq_to_net(cnn).get_feature_tensors()
        node_feats, edge_indices, edge_feats = sequential_to_feats(cnn)

        features.append((node_feats, edge_indices, edge_feats))
        accuracies.append(accuracy)
    if not os.path.exists('gmn_data'):
        os.makedirs('gmn_data')
    if os.path.exists('gmn_data/cnn_features.pt'):
        # load
        features = torch.load('gmn_data/cnn_features.pt') + features
        accuracies = torch.load('gmn_data/cnn_accuracies.pt') + accuracies
    torch.save(features, 'gmn_data/ccnn_features.pt')
    torch.save(accuracies, 'gmn_data/ccnn_accuracies.pt')

