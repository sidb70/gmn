import torch.nn as nn
import torch
import numpy as np
from argparse import ArgumentParser
import sys
import os
from utils import split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.mpnn_models import BaseMPNN
from preprocessing.data_loader import get_dataset
import time
from preprocessing.data_loader import get_dataset
from preprocessing.generate_data import generate_random_cnn
from gmn_lim.model_arch_graph import seq_to_feats

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, feats, labels, batch_size, criterion, optimizer):
    for i in range(0, len(feats), batch_size):
        outs = []
        for j in range(i, min(i + batch_size, len(feats))):
            node_feat, edge_index, edge_feat, _ = feats[j]
            node_feat, edge_index, edge_feat = (
                torch.tensor(node_feat).to(DEVICE),
                torch.tensor(edge_index).to(DEVICE),
                torch.tensor(edge_feat).to(DEVICE),
            )
            out = model.forward(node_feat, edge_index, edge_feat)
            outs.append(out)
        outs = torch.cat(outs, dim=1).squeeze(0).to(DEVICE)
        y = torch.tensor(labels[i : i + batch_size]).to(DEVICE)
        loss = criterion(outs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Predictions: ", outs)
        # print("Labels: ", y)


def eval_step(model, feats, labels, batch_size, criterion):
    model.eval()
    for i in range(0, len([feats]), batch_size):
        outs = []
        for j in range(i, min(i + batch_size, len(feats))):
            node_feat, edge_index, edge_feat, _ = feats[j]
            node_feat, edge_index, edge_feat = (
                torch.tensor(node_feat, dtype=torch.float32).to(DEVICE),
                torch.tensor(edge_index).to(DEVICE),
                torch.tensor(edge_feat, dtype=torch.float32).to(DEVICE),
            )
            out = model(node_feat, edge_index, edge_feat)
            outs.append(out)
    outs = torch.cat(outs, dim=1).squeeze(0).to(DEVICE)
    y = torch.tensor(labels[i : i + batch_size]).to(DEVICE)
    loss = criterion(outs, y)
    print("Loss: ", loss)
    print("Predictions: ", outs)
    print("Labels: ", y)


def train_mpnn(args):
    hidden_dim = args.hidden_dim
    feats_path = args.feats_path
    label_path = args.label_path
    batch_size = args.batch_size
    valid_size = args.valid_size
    test_size = args.test_size
    num_epochs = args.epochs
    lr = args.lr

    feats, labels = get_dataset(feats_path, label_path)

    train_set, valid_set, test_set = split(valid_size, test_size, feats, labels)
    model = BaseMPNN(hidden_dim).to(DEVICE)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    feats_train, labels_train = train_set
    feats_valid, labels_valid = valid_set
    feats_test, labels_test = test_set
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        print("\nTraining step")
        train_epoch(
            model,
            feats_train,
            labels_train,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=batch_size,
        )
        print("\nValidation step")
        eval_step(model, feats_valid, labels_valid, batch_size, criterion=criterion)
    # test
    print("\nTest step")
    # eval_step(model, feats_test, labels_test, batch_size, criterion=criterion)
    return model


def eval_model(args, model):
    feats_path = args.feats_path
    label_path = args.label_path
    valid_size = args.valid_size
    test_size = args.test_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr

    cnn = generate_random_cnn()
    node_feats, edge_indices, edge_feats = seq_to_feats(cnn)

    node_feat, edge_index, edge_feat = (
        torch.tensor(node_feats, dtype=torch.float32).to(DEVICE),
        torch.tensor(edge_indices).to(DEVICE),
        torch.tensor(edge_feats, dtype=torch.float32).to(DEVICE),
    )

    start_time = time.time()
    out = model(node_feat, edge_index, edge_feat)
    print("predicted accuracy", out)
    print("gmn forward pass time taken", time.time() - start_time)

    eval_one_cnn(cnn)


def eval_one_cnn(cnn):

    batch_size = 1

    train_size = 40000
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(
        root="./data/cifar10", train=True, download=True, transform=transform
    )
    train_sampler = SubsetRandomSampler(torch.randperm(len(trainset))[:train_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

    testset = CIFAR10(
        root="./data/cifar10", train=False, download=True, transform=transform
    )
    test_size = train_size // 4
    test_sampler = SubsetRandomSampler(torch.randperm(len(testset))[:test_size])

    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler)

    start_time = time.time()

    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = cnn(images).reshape(-1, 10)
            labels = labels.reshape(-1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total

    print("eval loop time taken", time.time() - start_time)

    print(f"\nAccuracy: {accuracy}")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--feats_path", type=str, default="./data/cnn/features.pt")
    args.add_argument("--label_path", type=str, default="./data/cnn/accuracies.pt")
    args.add_argument("--node_feat_dim", type=int, default=3)
    args.add_argument("--edge_feat_dim", type=int, default=6)
    args.add_argument("--node_hidden_dim", type=int, default=16)
    args.add_argument("--edge_hidden_dim", type=int, default=16)
    args.add_argument("--hidden_dim", type=int, default=8)
    args.add_argument("--batch_size", type=int, default=2)
    args.add_argument("--valid_size", type=float, default=0.2)
    args.add_argument("--test_size", type=float, default=0.1)
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("--lr", type=float, default=0.01)
    args = args.parse_args()
    model = train_mpnn(args)

    eval_model(args, model)
