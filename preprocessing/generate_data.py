import sys
import os

from typing import Tuple, List
import time
import concurrent.futures as cfutures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import torch.nn as nn
from torch import optim

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


def train_cifar_worker(
    architecture_id: int,
    hyperparams: Hyperparameters,
    random_cnn_config: RandCNNConfig,
    device: torch.device,
    save_result_callback: callable = lambda x: None,
) -> TrainedNNResult:
    """
    Generates and trains a random CNN on CIFAR10 data
    with the given hyperparameters, on the given CUDA device.
    """

    print(f"Training model {architecture_id} on device {device}")

    hpo_vec = hyperparams.to_vec()
    batch_size, lr, n_epochs, momentum = hpo_vec

    trainloader, validloader, testloader = get_cifar_data(
        data_dir="./data/", device=torch.device(device), batch_size=batch_size
    )

    cnn = generate_random_cnn(random_cnn_config).to(device)
    n_params = sum(p.numel() for p in cnn.parameters())

    print(f"model has {n_params} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn.parameters(), lr=lr)

    model_feats: List[NetFeatures] = [seq_to_feats(cnn)]
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
        model_feats.append(seq_to_feats(cnn))

        print(f"Model Epoch {j+1}/{n_epochs}, \ttrain loss: {epoch_train_loss:.3f}, \tval loss: {epoch_val_loss:.3f}", end="\r" if j < n_epochs - 1 else "\n")
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
        epoch_feats=model_feats,
        train_losses=train_losses,
        val_losses=val_losses,
        final_accuracy=accuracy,
        hpo_vec=hpo_vec,
        device=device,
    )

    return result


def train_random_cnns_with_hyperparams(
    hyperparams_list=[Hyperparameters()],
    random_cnn_config=RandCNNConfig(),
    save_result_callback: callable = lambda x: None,
):
    """
    Generates random CNN architectures and trains them on CIFAR10 data.
    Trains one CNN for each set of hyperparameters provided.

    Args:
    - hyperparams: List of Hyperparameters, the hyperparameters to use for training each model.
    - random_cnn_config: RandomCNNConfig, the configuration for generating random CNNs
    - save_data_callback: A callback that is called after every model finished training.
    """

    n_architectures = len(hyperparams_list)

    print(
        f"Training {len(hyperparams_list)} CNN(s) with hyperparameters {hyperparams_list}"
    )

    model_num = 0
    free_devices = DEVICES.copy()
    start_id = time.time()
    with EXECUTOR as executor:
        futures = set()
        while model_num < n_architectures:
            print("model_num", model_num)
            if len(futures) == 0:
                # For each free device, submit a new model to train
                for i, hyperparams in enumerate(
                    hyperparams_list[model_num : model_num + len(free_devices)]
                ):
                    futures.add(
                        executor.submit(
                            train_cifar_worker,
                            time.time() + model_num / 1000.0,
                            hyperparams,
                            random_cnn_config,
                            free_devices[i],
                            save_result_callback,
                        )
                    )
                    model_num += 1
            for future in cfutures.as_completed(list(futures)):
                result: TrainedNNResult = future.result()
                save_result_callback(result)
                futures.remove(future)
                print("Freed device", result.device)
                if model_num < n_architectures:
                    futures.add(
                        executor.submit(
                            train_cifar_worker,
                            start_id + model_num / 1000.0,
                            hyperparams_list[model_num],
                            random_cnn_config,
                            result.device,
                            save_result_callback,
                        )
                    )
                    model_num += 1


def train_random_cnns_random_hyperparams(
    n_architectures=10,
    random_hyperparams_config=RandHyperparamsConfig(),
    random_cnn_config=RandCNNConfig(),
    save_result_callback: callable = lambda x: None,
):
    """
    Generates and trains random CNNs on cifar10, using random hyperparameters.
    """

    hyperparams = [random_hyperparams_config.sample() for _ in range(n_architectures)]
    print(len(hyperparams), n_architectures, "a")

    train_random_cnns_with_hyperparams(
        hyperparams_list=hyperparams,
        random_cnn_config=random_cnn_config,
        save_result_callback=save_result_callback,
    )
