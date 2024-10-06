import torch.nn as nn
import torch
import random
import numpy as np
from argparse import ArgumentParser
import sys
import os
from .utils import split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from resources.dataset_clients import HPOExperimentClient, LocalFileClient
from models.mpnn_models import HPOMPNN
from preprocessing.data_loader import get_dataset
from preprocessing.generate_data import generate_random_cnn
from gmn_lim.model_arch_graph import seq_to_feats


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, feats, labels, batch_size, criterion, optimizer):
    epoch_running_loss =0 
    num_batches = 0
    for i in range(0, len(feats), batch_size):
        outs = []
        for j in range(i, min(i + batch_size, len(feats))):
            #print("j: ", type(feats[j][0]), type(feats[j][1]))
            (node_feat, edge_index, edge_feat), hpo_vec = feats[j]
            node_feat, edge_index, edge_feat, hpo_vec = (
                torch.tensor(node_feat).to(DEVICE),
                torch.tensor(edge_index).to(DEVICE),
                torch.tensor(edge_feat).to(DEVICE),
                torch.tensor(hpo_vec).to(DEVICE),
            )
            out = model.forward(node_feat, edge_index, edge_feat, hpo_vec)
            outs.append(out)
        outs = torch.cat(outs, dim=1).squeeze(0).to(DEVICE)
        y = torch.tensor(labels[i : i + batch_size]).to(DEVICE)
        loss = criterion(outs, y)
        epoch_running_loss += loss.item()
        num_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Loss: ", loss.item())
        # print("Predictions: ", outs)
        # print("Labels: ", y)
    print("Epoch Training Loss: ", epoch_running_loss / num_batches)


def eval_step(model, feats, labels, batch_size, criterion, hpo_grad_steps=75):
    model.eval()
    # freeze model
    for param in model.parameters():
        param.requires_grad = False

    val_running_loss = 0
    num_batches = 0
    for i in range(0, len(feats), batch_size):
        outs = []
        for j in range(i, min(i + batch_size, len(feats))):
            (node_feat, edge_index, edge_feat), hpo_vec = feats[j]
            node_feat, edge_index, edge_feat, hpo_vec = (
                torch.tensor(node_feat, dtype=torch.float32).to(DEVICE),
                torch.tensor(edge_index).to(DEVICE),
                torch.tensor(edge_feat, dtype=torch.float32).to(DEVICE),
                torch.tensor(hpo_vec).to(DEVICE),
            )
            # enable grad for hpo_vec
            # hpo_vec.requires_grad = True
            # for _ in range(hpo_grad_steps):
            #     out = model(node_feat, edge_index, edge_feat, hpo_vec)
            #     loss = criterion(out, torch.tensor(labels[j]).to(DEVICE))
            #     loss.backward()
            #     hpo_vec.data = hpo_vec.data - 0.01 * hpo_vec.grad
            #     hpo_vec.grad.zero_()
            with torch.no_grad():
                out = model(node_feat, edge_index, edge_feat, hpo_vec)
            outs.append(out)

        outs = torch.cat(outs, dim=1).squeeze(0).to(DEVICE)
        y = torch.tensor(labels[i : i + batch_size]).to(DEVICE)
        loss = criterion(outs, y)
        val_running_loss += loss.item()
        num_batches += 1
    print("Val Num Batches: ", num_batches, "Batch Size: ", batch_size)
    print("Val Loss: ", val_running_loss / num_batches)

    # unfreeze model
    for param in model.parameters():
        param.requires_grad = True


def train_hpo(args, feats, labels):
    results_dir = args.results_dir
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    valid_size = args.valid_size
    test_size = args.test_size
    num_epochs = args.epochs
    lr = args.lr

    os.makedirs(results_dir, exist_ok=True)
    # client = HPOExperimentClient(LocalFileClient('/mnt/home/bhatta70/Documents/gmn/gmn/data/cnn_hpo'))
    # dataset = client.read_dataset()
    # feats, labels = dataset

    # print("Feats: ", type(feats), "composed of", type(feats[0]))
    # print("Labels: ", type(labels), "composed of", type(labels[0]))

    train_set, valid_set, test_set = split( feats, labels, test_size, valid_size)
    model = HPOMPNN(hidden_dim, hpo_dim=len(feats[0][-1])).to(DEVICE)

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
    eval_step(model, feats_test, labels_test, batch_size, criterion=criterion)

    return model


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--results_dir", type=str, default="data/hpo_result", help="Directory to save results")
    args.add_argument("--node_feat_dim", type=int, default=3)
    args.add_argument("--edge_feat_dim", type=int, default=6)
    args.add_argument("--node_hidden_dim", type=int, default=16)
    args.add_argument("--edge_hidden_dim", type=int, default=16)
    args.add_argument("--hidden_dim", type=int, default=8)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--valid_size", type=float, default=0.1)
    args.add_argument("--test_size", type=float, default=0.1)
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--lr", type=float, default=0.01)
    args = args.parse_args()
    model = train_hpo(args)
