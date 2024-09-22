import sys
import os
import concurrent.futures as cfutures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .hpo_configs import Hyperparameters, RandHyperparamsConfig
from .get_cifar_data import get_cifar_data
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import Tuple
import torch
import torch.nn as nn
from torch import optim
from gmn_lim.model_arch_graph import seq_to_feats
from preprocessing.generate_nns import generate_random_cnn, RandCNNConfig


# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS > 0:  
    DEVICES = [torch.device(f'cuda:{i}') for i in range(NUM_GPUS)]
else:
    DEVICES = [torch.device('cpu')]

EXECUTOR = ThreadPoolExecutor(max_workers=len(DEVICES))
# EXECUTOR = ProcessPoolExecutor(max_workers=len(DEVICES))

print("Using devices", DEVICES)

def train_random_cnns_hyperparams(
    results_dir,
    save_data_callback: callable,
    random_cnn_config: RandCNNConfig,
    random_hyperparams_config: RandHyperparamsConfig,
    n_architectures=10,
):
    """
    Generates and trains random CNNs, using random hyperparameters.
    """
    hyperparams = [random_hyperparams_config.sample() for _ in range(n_architectures)]  
    print("Training with hyperparams", hyperparams)

    train_cnns_cifar10(
        results_dir='data/cnn_hpo/',
        hyperparams=hyperparams,
        random_cnn_config=random_cnn_config,
        save=False,
        save_data_callback=save_data_callback,
    )


def train_cifar_worker(
    architecture_id: int, 
    hyperparams: Hyperparameters, 
    random_cnn_config: RandCNNConfig, 
    device: torch.device
):
    """
    Generates and trains a random CNN on CIFAR10 data with the given hyperparameters, on the given CUDA device.

    Args:
    - architecture_id: int, just used for logging
    """

    print("Training model", architecture_id+1,  ' on device', device)
    hpo_vec = hyperparams.to_vec()
    batch_size, lr, n_epochs, momentum = hpo_vec

    trainloader, testloader = get_cifar_data(data_dir='./data/', device=torch.device(device), batch_size=batch_size)

    cnn = generate_random_cnn(random_cnn_config).to(device)
    n_params = sum(p.numel() for p in cnn.parameters())

    print("Worker", architecture_id, "has", n_params, "parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn.parameters(), lr=lr)

    running_losses = []
    batch_nums = []

    for j in range(n_epochs):
        running_loss = 0.0
        for k, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = cnn(inputs).reshape(-1, 10)
            labels = labels.reshape(-1)
            assert labels.shape[0] > 0
            assert torch.min(labels) >= 0 
            assert torch.max(labels) < 10
            try:
                loss = criterion(outputs, labels)
            except Exception as e:
                print(outputs, labels)
                raise e
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (k+1) % 50 == 0 or k == len(trainloader) - 1:
                running_loss_batches = 50 if (k+1) % 50 == 0 else (k+1) % 50

                avg_running_loss = running_loss / running_loss_batches
                print(
                    f"\rTraining one cnn. {n_params} params, Epoch {j+1}/{n_epochs}, Batch {k+1}/{len(trainloader)}, Running Loss: {avg_running_loss:.3f}",end=""
                )
                running_losses.append(avg_running_loss)
                batch_nums.append(j * len(trainloader) + k)

                running_loss = 0.0

    # calculate accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = cnn(images).reshape(-1, 10)
            labels = labels.reshape(-1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total

    print(f"\nAccuracy: {accuracy}")
    node_feats, edge_indices, edge_feats = seq_to_feats(cnn)
    features= (node_feats, edge_indices, edge_feats, hpo_vec)
    return features, accuracy, device


def train_cnns_cifar10(
    results_dir,
    hyperparams=[Hyperparameters()],
    random_cnn_config=RandCNNConfig(n_classes=10),
    save=True,
    replace_if_existing=False,
    save_data_callback=None,
):
    """
    Generates random CNN architectures and trains them on CIFAR10 data
    Saves the resulting CNN node and edge features and their accuracies in directory

    Args:
    - hyperparams: List of Hyperparameters, the hyperparameters to use for training each model.
    - random_cnn_config: RandomCNNConfig, the configuration for generating random CNNs
    - optimizer_type: torch.optim.Optimizer, the optimizer to use. If None, uses SGD
    - data_dir: str, the directory in which the CIFAR10 data is stored
    - results_dir: str, the directory in which to save the CNN features and their accuracies
    - save: bool, whether to save the results to results_dir
    - replace_if_existing: bool, whether to replace the existing results if they exist or append to them

    Saves these files to the specified directory:
    - {results_dir}/features.pt: 
        list of tuples (
            node_feats: n_nodes x 3 Tensor,
            edge_indices: n_params x 2 Tensor, 
            edge_feats: n_params x 6 Tensor,
            hpo_vec: list of hyperparameters
        ) for each model
    
    - {results_dir}/accuracies.pt: list of accuracies for each model.
    """

    n_architectures = len(hyperparams)

    print(f"Training {len(hyperparams)} cnn(s) with hyperparameters {hyperparams}")

    model_num=0
    free_devices = DEVICES.copy()
    with EXECUTOR as executor:
        futures = set()
        while model_num < n_architectures:
            if len(futures) == 0:
                for i, hpo_config in enumerate(hyperparams[model_num:model_num+len(free_devices)]):
                    futures.add(executor.submit(train_cifar_worker, model_num, hpo_config, random_cnn_config, free_devices[i]))
                    model_num += 1
            for future in cfutures.as_completed(list(futures)):
                feature, accuracy, free_device = future.result()
                futures.remove(future)
                print("Freed device", free_device)
                save_data_callback(feature, accuracy)
                if model_num < n_architectures:
                    futures.add(executor.submit(train_cifar_worker, model_num, hyperparams[model_num], random_cnn_config, free_device))       
                    model_num += 1

