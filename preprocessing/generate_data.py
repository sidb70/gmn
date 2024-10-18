import sys
import os
import random
from typing import Tuple, List, Callable
import uuid
import concurrent.futures as cfutures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gmn_lim.model_arch_graph import seq_to_feats
from preprocessing.get_cifar_data import get_cifar_data
from preprocessing.generate_nns import generate_random_cnn, RandCNNConfig
from preprocessing.preprocessing_types import *


# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS > 0:
    DEVICES = [torch.device(f"cuda:{i}") for i in range(NUM_GPUS)]
else:
    DEVICES = [torch.device("cpu")]

EXECUTOR = ThreadPoolExecutor(max_workers=len(DEVICES))

print("Using devices", DEVICES)


def train_random_cnn(
    architecture_id: str,
    hyperparams: Hyperparameters,
    random_cnn_config: RandCNNConfig,
    device: torch.device,
) -> TrainedNNResult:
    """
    Generates and trains a random CNN on CIFAR10 data
    with the given hyperparameters, on the given CUDA device.
    """

    # print(f"Training model {architecture_id} on device {device}")

    hpo_vec = hyperparams.to_vec()
    batch_size, lr, n_epochs, momentum = hpo_vec

    trainloader, validloader, testloader = get_cifar_data(
        data_dir="./data/", device=torch.device(device), batch_size=batch_size
    )

    cnn = generate_random_cnn(random_cnn_config).to(device)
    # n_params = sum(p.numel() for p in cnn.parameters())

    # print(f"model has {n_params} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn.parameters(), lr=lr)

    epoch_feats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
        seq_to_feats(cnn)
    ]
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
                if (
                    labels.shape[0] == 0
                    or torch.min(labels) < 0
                    or torch.max(labels) >= 10
                ):
                    print("Invalid labels", labels.shape, labels)
                assert labels.shape[0] > 0
                assert torch.min(labels) >= 0
                assert torch.max(labels) < 10
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        epoch_train_loss = running_train_loss / len(trainloader)
        epoch_val_loss = running_val_loss / len(validloader)
        train_losses.append(running_train_loss / len(trainloader))
        val_losses.append(running_val_loss / len(validloader))
        epoch_feats.append(seq_to_feats(cnn))
        print(
            f"Epoch {j+1}/{n_epochs}, \ttrain loss: {epoch_train_loss:.3f}, \tval loss: {epoch_val_loss:.3f}",
            end="\r" if j < n_epochs - 1 else "\n",
        )
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

    result = TrainedNNResult(
        model_id=architecture_id,
        torch_model=cnn,
        epoch_feats=epoch_feats,
        train_losses=train_losses,
        val_losses=val_losses,
        final_accuracy=accuracy,
        hpo_vec=hpo_vec,
        device=device,
    )

    return result


def train_random_cnns_with_hyperparams(
    device: torch.device,
    save_result_callback: Callable[[TrainedNNResult], None],
    hyperparams_list: List[Hyperparameters],
    random_cnn_config: RandCNNConfig,
):
    """
    trains all of the models in hyperparams_list on the given device
    """
    print(f"Training {len(hyperparams_list)} CNN(s) on device {device}")

    for hyperparams in tqdm(
        hyperparams_list, desc=f"Models trained on device {device}"
    ):
        model_id = str(uuid.uuid4())
        try:
            result = train_random_cnn(model_id, hyperparams, random_cnn_config, device)
        except AssertionError as e:
            print(f"Error training model with hyperparameters {hyperparams}: {e}")
            continue
        save_result_callback(result)
    print("Finished training on device", device)


def train_random_cnns_random_hyperparams(
    save_result_callback: Callable[[TrainedNNResult], None],
    n_architectures,
    random_hyperparams_config=RandHyperparamsConfig(),
    random_cnn_config=RandCNNConfig(),
):
    """
    Generates random CNN architectures and trains them on CIFAR10 data.
    Trains one CNN for each set of hyperparameters provided.

    Args:
    - random_cnn_config: RandomCNNConfig, the configuration for generating random CNNs
    - save_result_callback: A callback that is called after every model finished training.
    """

    hyperparams_list = [
        random_hyperparams_config.sample() for _ in range(n_architectures)
    ]
    free_devices = DEVICES.copy()
    with EXECUTOR as executor:
        futures = []
        num_gpus = len(free_devices)
        partitions = [hyperparams_list[i::num_gpus] for i in range(num_gpus)]

        for i, device in enumerate(free_devices):
            hyperparams_list_device = list(partitions[i])
            futures.append(
                executor.submit(
                    train_random_cnns_with_hyperparams,
                    device,
                    save_result_callback,
                    hyperparams_list_device,
                    random_cnn_config,
                )
            )
        for future in cfutures.as_completed(futures):
            future.result()
