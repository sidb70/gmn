import os
import sys
import torch
import torch.nn as nn
from torch import optim
import torchvision
from .write_ffcv_data import cifar_10_to_beton
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice,  RandomHorizontalFlip, RandomTranslate, Cutout, ToTorchImage, Convert, Squeeze
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.fields.basics import Operation
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.generate_nns import generate_random_cnn
from param_graph.gmn_lim.model_arch_graph import seq_to_feats


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_random_cnns(
    n_architectures=10,
    batch_size=512,
    n_epochs=2,
    lr=0.001, 
    momentum=0.9,
    num_workers=4,
    directory='data/',
    replace_if_existing=False,
    **random_cnn_kwargs
):
    """
    Generates random CNN architectures and trains them on CIFAR10 data

    Args:
    - n_architectures: int, the number of architectures to generate
    - n_epochs: int, the number of epochs to train each architecture
    - random_cnn_kwargs: see generate_random_cnn

    Saves these files to the specified directory:
    - features.pt: list of tuples (node_feats, edge_indices, edge_feats) for each model
    - accuracies.pt: list of accuracies for each model. 
    """

    torch.manual_seed(0)

    cifar_10_dir = os.path.join(directory, 'cifar10')
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
        loaders[name] = Loader(os.path.join(directory,'cifar10', f'cifar_{name}.beton'),
                                batch_size=batch_size,
                                num_workers=num_workers,
                                order=OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline,
                                        'label': label_pipeline})
    ###
    trainloader = loaders['train']
    testloader = loaders['test']
    features = []
    accuracies = []

    for i in range(n_architectures):
        cnn = generate_random_cnn(**random_cnn_kwargs).to(DEVICE)

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
                if k % 1 == 0:
                    print(f'\rTraining model {i+1}/{n_architectures}, Epoch {j+1}/{n_epochs}, Batch {k+1}/{len(trainloader)}, Loss: {running_loss/2000:.3f}', end='')
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

        # print('total params', sum(p.numel() for p in cnn.parameters()))
        # print(cnn)
        node_feats, edge_indices, edge_feats = seq_to_feats(cnn)

        # print(edge_indices.shape, edge_feats.shape)

        features.append((node_feats, edge_indices, edge_feats))
        accuracies.append(accuracy)

    print('saving data')

    # ensure that the directory is valid and exists
    save_dir = os.path.join(directory, 'cnn')
    os.makedirs(save_dir, exist_ok=True)

    if replace_if_existing:
        features_path = os.path.join(save_dir, 'features.pt')
        accuracies_path = os.path.join(save_dir, 'accuracies.pt')
        if os.path.exists(features_path):
            os.remove(features_path)
        if os.path.exists(accuracies_path):
            os.remove(accuracies_path)

    else:
        if os.path.exists(os.path.join(save_dir, 'features.pt')) and os.path.exists(os.path.join(save_dir, 'accuracies.pt')):
            # load
            features = torch.load(os.path.join(save_dir, 'features.pt')) + features
            accuracies = torch.load(os.path.join(save_dir, 'accuracies.pt')) + accuracies

    torch.save(features, os.path.join(save_dir, 'features.pt'))
    torch.save(accuracies, os.path.join(save_dir, 'accuracies.pt'))

