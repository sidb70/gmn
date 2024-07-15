from models.models import BaseMPNN
from preprocessing.data_loader import get_dataset
import torch.nn as nn
import torch
import random
import numpy as np
from argparse import ArgumentParser
torch.manual_seed(0)
random.seed(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split(valid_size: float, test_size: float,  node_feats, edge_indices, edge_feats, labels):
    n = len(node_feats)
    indices = np.arange(n)
    random.shuffle(indices)
    num_valid = int(n*valid_size)
    num_test = int(n*test_size)
    num_train = int(n - (num_valid + num_test))
    assert num_train>0, 'be nice and leave something to train on'

    train_indices = indices[:num_train]
    valid_indices = indices[num_train:num_train + num_valid]
    test_indices = indices[num_train + num_valid:]

    node_feats_train = [node_feats[i] for i in train_indices]
    edge_indices_train = [edge_indices[i] for i in train_indices]
    edge_feats_train = [edge_feats[i] for i in train_indices]
    labels_train = [labels[i] for i in train_indices]
    node_feats_valid = [node_feats[i] for i in valid_indices]
    edge_indices_valid = [edge_indices[i] for i in valid_indices]
    edge_feats_valid = [edge_feats[i] for i in valid_indices]
    labels_valid = [labels[i] for i in valid_indices]
    node_feats_test = [node_feats[i] for i in test_indices]
    edge_indices_test = [edge_indices[i] for i in test_indices]
    edge_feats_test = [edge_feats[i] for i in test_indices]
    labels_test = [labels[i] for i in test_indices]
    train_set = [node_feats_train, edge_indices_train, edge_feats_train, labels_train]
    valid_set = [node_feats_valid, edge_indices_valid, edge_feats_valid, labels_valid]
    test_set = [node_feats_test, edge_indices_test, edge_feats_test, labels_test]
    return train_set, valid_set, test_set
def train_epoch(model, node_feats_train, edge_indices_train, edge_feats_train, labels_train , batch_size, criterion, optimizer):
    for i in range(0, len(node_feats_train), batch_size):
        outs = []
        for j in range(i, min(i+batch_size, len(node_feats_train))):
            node_feat = node_feats_train[j].to(DEVICE)
            edge_index = edge_indices_train[j].clone().to(DEVICE)
            edge_feat = edge_feats_train[j].clone().to(DEVICE)
            out = model.forward(node_feat, edge_index, edge_feat)
            outs.append(out)
        outs = torch.cat(outs, dim=1).squeeze(0).to(DEVICE)
        y = torch.tensor(labels_train[i:i+batch_size]).to(DEVICE)
        loss = criterion(outs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Predictions: ", outs)
        # print("Labels: ", y)
def eval_step(model, node_feats, edge_indices, edge_feats, labels, criterion, batch_size):
    model.eval()
    for i in range(0, len([node_feats]), batch_size):
        outs = []
        for j in range(i, min(i+batch_size, len(node_feats))):
            node_feat = node_feats[j].clone().to(DEVICE)
            edge_index = edge_indices[j].clone().to(DEVICE)
            edge_feat = edge_feats[j].clone().to(DEVICE)
            out = model(node_feat, edge_index, edge_feat)
            outs.append(out)
    outs = torch.cat(outs).squeeze(0).to(DEVICE)
    y = torch.tensor(labels[i:i+batch_size]).to(DEVICE)
    loss = criterion(outs, y)
    print("Loss: ", loss)
    print("Predictions: ", outs)
    print("Labels: ", y)

def train(args):
    node_feat_dim = args.node_feat_dim
    edge_feat_dim = args.edge_feat_dim
    node_hidden_dim = args.node_hidden_dim
    edge_hidden_dim = args.edge_hidden_dim
    batch_size = args.batch_size
    valid_size = args.valid_size
    test_size = args.test_size
    num_epochs = args.epochs
    lr = args.lr

    node_feats, edge_indices, edge_feats, labels = get_dataset(node_feats_path=None, 
                                                               edge_indices_path=None, 
                                                               edge_feats_path=None, 
                                                               labels_path=None)
    
    train_set, valid_set, test_set = split(valid_size, test_size, node_feats, edge_indices, edge_feats, labels)
    model = BaseMPNN(node_feat_dim, edge_feat_dim, node_hidden_dim, edge_hidden_dim).to(DEVICE)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    node_feats_train, edge_indices_train, edge_feats_train, labels_train = train_set
    node_feats_valid, edge_indices_valid, edge_feats_valid, labels_valid = valid_set
    node_feats_test, edge_indices_test, edge_feats_test, labels_test = test_set
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        print("\nTraining step")
        train_epoch(model,  node_feats_train, edge_indices_train, edge_feats_train, labels_train, 
                    criterion=criterion, optimizer=optimizer, batch_size=batch_size)
        print("\nValidation step")
        eval_step(model, node_feats_valid, edge_indices_valid, edge_feats_valid, labels_valid, criterion, batch_size)
    # test
    print('\nTest step')
    eval_step(model, node_feats_test, edge_indices_test, edge_feats_test, labels_test, criterion, batch_size)
    return model

if __name__=='__main__':
    args = ArgumentParser()
    args.add_argument('--node_feat_dim', type=int, default=3)
    args.add_argument('--edge_feat_dim', type=int, default=6)
    args.add_argument('--node_hidden_dim', type=int, default=16)
    args.add_argument('--edge_hidden_dim', type=int, default=16)
    args.add_argument('--batch_size', type=int, default=2)
    args.add_argument('--valid_size', type=float, default=0.34)
    args.add_argument('--test_size', type=float, default=0.34)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--lr', type=float, default=0.01)
    args = args.parse_args()
    train(args)

