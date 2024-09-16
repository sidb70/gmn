import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from preprocessing.generate_data import generate_random_cnn, RandHyperparamsConfig, RandCNNConfig
from preprocessing.generate_data import get_cifar_data
import matplotlib.pyplot as plt


"""
Sequential(
  (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1))
  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
  (3): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1))
  (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU()
  (6): AdaptiveAvgPool2d(output_size=(1, 1))
  (7): Flatten()
  (8): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  (9): Linear(in_features=64, out_features=128, bias=True)
  (10): ReLU()
  (11): Linear(in_features=128, out_features=64, bias=True)
  (12): ReLU()
  (13): Linear(in_features=64, out_features=10, bias=True)
  (14): ReLU()
)
"""

def train_1_cnn():

    # train the cnn, plot train and validation loss curve.

    # copied from generate_data::train_cifar_worker

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    np.random.seed(0)
    hyperparams = RandHyperparamsConfig().sample()
    random_cnn_config = RandCNNConfig(
        n_conv_layers_range=(3,5),
        n_fc_layers_range=(3,5),
        pool_after_conv=True,
    )

    hpo_vec = hyperparams.to_vec()
    batch_size, lr, n_epochs, momentum = hpo_vec

    print(hpo_vec)

    n_epochs=10
    batch_size=16
    lr = 0.0005

    trainloader, testloader = get_cifar_data(data_dir='./data/', device=torch.device(device), batch_size=batch_size)

    cnn = generate_random_cnn(random_cnn_config).to(device)
    print(cnn)
    n_params = sum(p.numel() for p in cnn.parameters())

    # exit(0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn.parameters(), lr=lr)

    running_losses = []
    batch_nums = []

    for j in range(n_epochs):
        running_loss = 0.0
        # running_loss_batches = 0
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
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (k+1) % 50 == 0 or k == len(trainloader) - 1:
                running_loss_batches = 50 if (k+1) % 50 == 0 else (k+1) % 50

                avg_running_loss = running_loss / running_loss_batches
                print(
                    f"\rTraining one cnn. {n_params} params, Epoch {j+1}/{n_epochs}, Batch {k+1}/{len(trainloader)}, Running Loss: {avg_running_loss:.3f}",end=""
                )
                running_losses.append(avg_running_loss)
                batch_nums.append(j * len(trainloader) + k)

                running_loss = 0.0

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

        print(f"\nAccuracy: {correct / total}")

    print("Final loss:", running_losses[-1])
    epoch_nums = np.array(batch_nums) / len(trainloader)

    # plot losses and save fig to png

    plt.plot(epoch_nums, running_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"n params: {n_params}, batch size: {batch_size}, lr: {lr}, n epochs: {n_epochs}")
    plt.savefig("../data/train_loss_curve.png")



if __name__ == "__main__":
    train_1_cnn()