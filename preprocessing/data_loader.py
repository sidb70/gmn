import torch
import torch.nn as nn
from argparse import ArgumentParser


def get_dataset(feats_path: str, labels_path: str) -> torch.Tensor:
    # load feature matrices and labels
    return torch.load(feats_path), torch.load(labels_path)

    # mock data
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
        nn.Conv2d(3, 6, 3),
        nn.ReLU(),
        nn.Conv2d(6, 4, 3),
        nn.BatchNorm2d(4),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(18, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
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


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--feats_path', type=str,
                      default='gmn_data/cnn_features.pt')
    args.add_argument('--labels_path', type=str,
                      default='gmn_data/cnn_accuracies.pt')
    args = args.parse_args()
    node_feats, edge_indices, edge_feats, labels = get_dataset(
        args.feats_path, args.labels_path)
    print("node_feats: ", len(node_feats),
          ' elem 0 shape', node_feats[0].shape)
    print("edge_indices: ", len(edge_indices),
          ' elem 0 shape', edge_indices[0].shape)
    print("edge_feats: ", len(edge_feats),
          ' elem 0 shape', edge_feats[0].shape)
