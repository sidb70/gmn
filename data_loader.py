import torch
import torch.nn as nn
from argparse import ArgumentParser
from param_graph.seq_to_net import seq_to_net

def get_dataset(args):
    # load feature matrices and labels
    # node_feats = torch.load(args.node_feats_path)
    # edge_indices = torch.load(args.edge_indices_path)
    # edge_feats = torch.load(args.edge_feats_path)
    # labels = torch.load(args.labels_path)

    ## mock data
    nn1 = nn.Sequential(
        nn.Conv2d(3, 4, 5),
        nn.ReLU(),
        nn.Conv2d(4, 6, 5),
        nn.BatchNorm2d(6),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(6, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    param_graph_1 = seq_to_net(nn1)
    nn2 = nn.Sequential(
        nn.Conv2d(3, 4, 5),
        nn.ReLU(),
        nn.Conv2d(4, 6, 5),
        nn.BatchNorm2d(6),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(6, 4),
        nn.ReLU(),
        nn.Linear(4, 2)
    )
    param_graph_2 = seq_to_net(nn2)
    node_feats1, edge_indices1, edge_feats1 = param_graph_1.get_feature_tensors()
    node_feats2, edge_indices2, edge_feats2 = param_graph_2.get_feature_tensors()
    node_feats = torch.cat([node_feats1, node_feats2], dim=0)
    
    max_edge_indices = max(edge_indices1.shape[1], edge_indices2.shape[1])
    padded_edge_indices1 = -1 * torch.ones(2, max_edge_indices - edge_indices1.shape[1])
    padded_edge_indices2 = -1 * torch.ones(2, max_edge_indices - edge_indices2.shape[1])
    edge_indices1 = torch.cat([edge_indices1, padded_edge_indices1], dim=1)
    edge_indices2 = torch.cat([edge_indices2, padded_edge_indices2], dim=1)

    edge_indices = torch.cat([edge_indices1, edge_indices2], dim=0)
    edge_feats = torch.cat([edge_feats1, edge_feats2], dim=0)
    labels = torch.tensor([.89, .93])
    return node_feats, edge_indices, edge_feats, labels
if __name__=='__main__':
    args = ArgumentParser()
    args.add_argument('--node_feats_path', type=str, default='data/node_feats.pt')
    args.add_argument('--edge_indices_path', type=str, default='data/edge_indices.pt')
    args.add_argument('--edge_feats_path', type=str, default='data/edge_feats.pt')
    args.add_argument('--labels_path', type=str, default='data/labels.pt')
    args = args.parse_args()
    dataset = get_dataset(args)
    node_feats, edge_indices, edge_feats, labels = dataset
    print("node_feats: ", node_feats.shape)
    print("edge_indices: ", edge_indices.shape)
    print("edge_feats: ", edge_feats.shape)