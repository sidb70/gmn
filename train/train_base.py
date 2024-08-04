import torch.nn as nn
import torch
import random
import numpy as np
from argparse import ArgumentParser
import sys
import os
from utils import split
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.models import BaseMPNN
from preprocessing.data_loader import get_dataset


torch.manual_seed(0)
random.seed(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, feats, labels , batch_size, criterion, optimizer):
    for i in range(0, len(feats), batch_size):
        outs = []
        for j in range(i, min(i+batch_size, len(feats))):
            node_feat, edge_index, edge_feat = feats[j]
            node_feat, edge_index, edge_feat = torch.tensor(node_feat).to(DEVICE), torch.tensor(edge_index).to(DEVICE), torch.tensor(edge_feat).to(DEVICE)
            out = model.forward(node_feat, edge_index, edge_feat)
            outs.append(out)
        outs = torch.cat(outs, dim=1).squeeze(0).to(DEVICE)
        y = torch.tensor(labels[i:i+batch_size]).to(DEVICE)
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
        for j in range(i, min(i+batch_size, len(feats))):
            node_feat, edge_index, edge_feat = feats[j]
            node_feat, edge_index, edge_feat = torch.tensor(node_feat,dtype=torch.float32).to(DEVICE), \
                                               torch.tensor(edge_index).to(DEVICE), \
                                               torch.tensor(edge_feat,dtype=torch.float32).to(DEVICE)
            out = model(node_feat, edge_index, edge_feat)
            outs.append(out)
    outs = torch.cat(outs, dim=1).squeeze(0).to(DEVICE)
    y = torch.tensor(labels[i:i+batch_size]).to(DEVICE)
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
        train_epoch(model, feats_train, labels_train, 
                    criterion=criterion, optimizer=optimizer, batch_size=batch_size)
        print("\nValidation step")
        eval_step(model, feats_valid, labels_valid, batch_size, criterion=criterion)
    # test
    print('\nTest step')
    eval_step(model, feats_test, labels_test, batch_size, criterion=criterion)
    return model


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--feats_path', type=str, default = './data/cnn/features.pt')
    args.add_argument('--label_path', type=str, default= './data/cnn/accuracies.pt')
    args.add_argument('--node_feat_dim', type=int, default=3)
    args.add_argument('--edge_feat_dim', type=int, default=6)
    args.add_argument('--node_hidden_dim', type=int, default=16)
    args.add_argument('--edge_hidden_dim', type=int, default=16)
    args.add_argument('--hidden_dim', type=int, default=8)
    args.add_argument('--batch_size', type=int, default=2)
    args.add_argument('--valid_size', type=float, default=0.2)
    args.add_argument('--test_size', type=float, default=0.1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--lr', type=float, default=0.01)
    args = args.parse_args()
    train_mpnn(args)