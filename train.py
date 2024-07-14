from models.models import BaseMPNN
from data_loader import get_dataset
import torch.nn as nn
import torch
import random
import numpy as np
from argparse import ArgumentParser
torch.manual_seed(0)
random.seed(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split(test_size: float, node_feats, edge_indices, edge_feats, labels):
    n = len(node_feats)
    indices = np.arange(n)
    random.shuffle(indices)
    split = int(n*test_size)
    test_indices = indices[:split]
    train_indices = indices[split:]
    node_feats_train = [node_feats[i] for i in train_indices]
    edge_indices_train = [edge_indices[i] for i in train_indices]
    edge_feats_train = [edge_feats[i] for i in train_indices]
    labels_train = [labels[i] for i in train_indices]
    node_feats_test = [node_feats[i] for i in test_indices]
    edge_indices_test = [edge_indices[i] for i in test_indices]
    edge_feats_test = [edge_feats[i] for i in test_indices]
    labels_test = [labels[i] for i in test_indices]
    return node_feats_train, edge_indices_train, edge_feats_train, labels_train, node_feats_test, edge_indices_test, edge_feats_test, labels_test
def train(args):
    node_feat_dim = args.node_feat_dim
    edge_feat_dim = args.edge_feat_dim
    node_hidden_dim = args.node_hidden_dim
    edge_hidden_dim = args.edge_hidden_dim
    batch_size = args.batch_size
    test_size = args.test_size
    num_epochs = args.epochs
    lr = args.lr

    node_feats, edge_indices, edge_feats, labels = get_dataset(node_feats_path=None, 
                                                               edge_indices_path=None, 
                                                               edge_feats_path=None, 
                                                               labels_path=None)
    
    node_feats_train, edge_indices_train, edge_feats_train, \
        labels_train, node_feats_test, edge_indices_test, edge_feats_test, labels_test \
            = split(test_size, node_feats, edge_indices, edge_feats, labels)
    model = BaseMPNN(node_feat_dim, edge_feat_dim, node_hidden_dim, edge_hidden_dim)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        for i in range(0, len(node_feats_train), batch_size):
            outs = []
            for j in range(i, min(i+batch_size, len(node_feats_train))):
                node_feat = node_feats_train[j]
                edge_index = edge_indices_train[j]
                edge_feat = edge_feats_train[j]
                out = model(node_feat, edge_index, edge_feat)
                outs.append(out)
            outs = torch.cat(outs, dim=1).squeeze(0).to(DEVICE)
            y = torch.tensor(labels_train[i:i+batch_size]).to(DEVICE)
            loss = criterion(outs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Predictions: ", outs)
            print("Labels: ", y)
            
        print("Epoch: ", epoch)

    # test
    for i in range(0, len([node_feats_test]), batch_size):
        outs = []
        for j in range(i, min(i+batch_size, len(node_feats_test))):
            node_feat = node_feats_test[j]
            edge_index = edge_indices_test[j]
            edge_feat = edge_feats_test[j]
            out = model(node_feat, edge_index, edge_feat)
            outs.append(out)
        outs = torch.cat(outs).squeeze(0).to(DEVICE)
        y = torch.tensor(labels_test[i:i+batch_size]).to(DEVICE)
        loss = criterion(outs, y)
        print("\nTest loss: ", loss)
        print("Predictions: ", outs)
        print("Labels: ", y)
        
    return model

if __name__=='__main__':
    args = ArgumentParser()
    args.add_argument('--node_feat_dim', type=int, default=3)
    args.add_argument('--edge_feat_dim', type=int, default=6)
    args.add_argument('--node_hidden_dim', type=int, default=16)
    args.add_argument('--edge_hidden_dim', type=int, default=16)
    args.add_argument('--batch_size', type=int, default=2)
    args.add_argument('--test_size', type=float, default=0.34)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--lr', type=float, default=0.01)
    args = args.parse_args()
    train(args)

