import sys
import os
import time
import concurrent.futures as cfutures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .preprocessing_types import Hyperparameters, RandHyperparamsConfig
from .get_cifar_data import get_cifar_data

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import Tuple
import torch
import torch.nn as nn
from torch import optim
from gmn_lim.model_arch_graph import seq_to_feats
from preprocessing.generate_nns import generate_random_cnn, RandCNNConfig
from preprocessing.preprocessing_types import HPOFeatures


# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS > 0:
    DEVICES = [torch.device(f"cuda:{i}") for i in range(NUM_GPUS)]
else:
    DEVICES = [torch.device("cpu")]

EXECUTOR = ThreadPoolExecutor(max_workers=len(DEVICES))
# EXECUTOR = ProcessPoolExecutor(max_workers=len(DEVICES))

print("Using devices", DEVICES)


def train_random_cnns_hyperparams(
    random_cnn_config: RandCNNConfig,
    random_hyperparams_config: RandHyperparamsConfig,
    n_architectures=10,
    save_data_callback: callable = lambda x: None,
):
    """
    Generates and trains random CNNs, using random hyperparameters.
    """

    hyperparams = [random_hyperparams_config.sample() for _ in range(n_architectures)]
    print("Training with hyperparams", hyperparams)

    train_random_cnns_cifar10(
        hyperparams=hyperparams,
        random_cnn_config=random_cnn_config,
        save_data_callback=save_data_callback,
    )


def train_cifar_worker(
    architecture_id: int,
    hyperparams: Hyperparameters,
    random_cnn_config: RandCNNConfig,
    device: torch.device,
    save_data_callback: callable = lambda x: None,
) -> Tuple[HPOFeatures, torch.Tensor, torch.device, int]:
    """
    Generates and trains a random CNN on CIFAR10 data with the given hyperparameters, on the given CUDA device.

    Args:
    - architecture_id: int, just used for logging
    - device: torch.device, the device the model is trained on
    """
    print("Training model", architecture_id + 1, " on device", device)
    hpo_vec = hyperparams.to_vec()
    batch_size, lr, n_epochs, momentum = hpo_vec

    trainloader, validloader, testloader = get_cifar_data(
        data_dir="./data/", device=torch.device(device), batch_size=batch_size
    )

    cnn = generate_random_cnn(random_cnn_config).to(device)
    n_params = sum(p.numel() for p in cnn.parameters())

    print("Worker", architecture_id, "has", n_params, "parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn.parameters(), lr=lr)

    batch_nums = []

    model_feats = [ seq_to_feats(cnn) ]
    train_losses = []
    val_losses = []
    

    for j in range(n_epochs):
        running_train_loss = 0.0
        for k, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = cnn(inputs).reshape(-1, 10)
            labels = labels.reshape(-1)
            assert labels.shape[0] > 0
            assert torch.min(labels) >= 0
            assert torch.max(labels) < 10
            loss = criterion(outputs, labels)
            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        running_val_loss = 0.0
        with torch.no_grad():
            for k, data in enumerate(validloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = cnn(inputs).reshape(-1, 10)
                labels = labels.reshape(-1)
                if labels.shape[0] == 0 or torch.min(labels) < 0 or torch.max(labels) >= 10:
                    print("Invalid labels", labels.shape, labels)
                assert labels.shape[0] > 0
                assert torch.min(labels) >= 0
                assert torch.max(labels) < 10
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        epoch_train_loss = running_train_loss/len(trainloader)
        epoch_val_loss = running_val_loss/len(validloader)
        train_losses.append(running_train_loss/len(trainloader))
        val_losses.append(running_val_loss/len(validloader))
        model_feats.append(seq_to_feats(cnn))

        print("Epoch", j, "train loss:", epoch_train_loss, "val loss:", epoch_val_loss, end="\r")
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
    # node_feats, edge_indices, edge_feats = seq_to_feats(cnn)
    # features = (node_feats, edge_indices, edge_feats, hpo_vec)
    return model_feats, train_losses, val_losses, accuracy, device, architecture_id


def train_random_cnns_cifar10(
    hyperparams=[Hyperparameters()],
    random_cnn_config=RandCNNConfig(n_classes=10),
    save_data_callback: callable = lambda x: None,
):
    """
    Generates random CNN architectures and trains them on CIFAR10 data
    Saves the resulting CNN node and edge features and their accuracies in directory

    Args:
    - hyperparams: List of Hyperparameters, the hyperparameters to use for training each model.
    - random_cnn_config: RandomCNNConfig, the configuration for generating random CNNs
    - save_data_callback: A callback that is called after every model finished training.
        the callback gets these arguments:
        - features: HPOFeatures
        - accuracy: float
    """

    n_architectures = len(hyperparams)

    print(f"Training {len(hyperparams)} cnn(s) with hyperparameters {hyperparams}")

    model_num = 0
    free_devices = DEVICES.copy()
    start_id = time.time()
    with EXECUTOR as executor:
        futures = set()
        while model_num < n_architectures:
            if len(futures) == 0:
                for i, hpo_config in enumerate(
                    hyperparams[model_num : model_num + len(free_devices)]
                ):
                    futures.add(
                        executor.submit(
                            train_cifar_worker,
                            time.time() + model_num/1000.,
                            hpo_config,
                            random_cnn_config,
                            free_devices[i],
                        )
                    )
                    model_num += 1
            for future in cfutures.as_completed(list(futures)):
                model_feats, train_losses, val_losses, accuracy, free_device, finished_model_id = future.result()
                futures.remove(future)
                print("Freed device", free_device)
                save_data_callback(model_feats, train_losses, val_losses,accuracy, finished_model_id)
                if model_num < n_architectures:
                    futures.add(
                        executor.submit(
                            train_cifar_worker,
                            start_id + model_num/1000.,
                            hyperparams[model_num],
                            random_cnn_config,
                            free_device,
                        )
                    )
                    model_num += 1
