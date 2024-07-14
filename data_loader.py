import torch
import torch.nn as nn
from argparse import ArgumentParser
from param_graph.seq_to_net import seq_to_net

def get_dataset(node_feats_path: str, edge_indices_path: str, edge_feats_path: str, labels_path: str) -> torch.Tensor:
    # load feature matrices and labels
    # node_feats = torch.load(node_feats_path)
    # edge_indices = torch.load(edge_indices_path)
    # edge_feats = torch.load(edge_feats_path)
    # labels = torch.load(labels_path)

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
        nn.Linear(4, 1)
    )
    param_graph_2 = seq_to_net(nn2)

    nn3 = nn.Sequential(
        nn.Conv2d(3,6,3),
        nn.ReLU(),
        nn.Conv2d(6,4,3),
        nn.BatchNorm2d(4),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(18,8),
        nn.ReLU(),
        nn.Linear(8,1)
    )
    param_graph_3 = seq_to_net(nn3)
    node_feats1, edge_indices1, edge_feats1 = param_graph_1.get_feature_tensors()
    node_feats2, edge_indices2, edge_feats2 = param_graph_2.get_feature_tensors()
    node_feats3, edge_indices3, edge_feats3 = param_graph_3.get_feature_tensors()

    node_feats = [node_feats1, node_feats2, node_feats3]
    edge_indices = [edge_indices1, edge_indices2, edge_indices3]
    edge_feats = [edge_feats1, edge_feats2, edge_feats3]
    labels = [.89, .93, .95]
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