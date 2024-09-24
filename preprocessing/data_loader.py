import torch
import torch.nn as nn
from argparse import ArgumentParser
from preprocessing.types import HPODataset


def get_dataset(feats_path: str, labels_path: str) -> HPODataset:
    # load feature matrices and labels
    return torch.load(feats_path), torch.load(labels_path)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--feats_path", type=str, default="gmn_data/cnn_features.pt")
    args.add_argument("--labels_path", type=str, default="gmn_data/cnn_accuracies.pt")
    args = args.parse_args()
    node_feats, edge_indices, edge_feats, labels = get_dataset(
        args.feats_path, args.labels_path
    )
    print("node_feats: ", len(node_feats), " elem 0 shape", node_feats[0].shape)
    print("edge_indices: ", len(edge_indices), " elem 0 shape", edge_indices[0].shape)
    print("edge_feats: ", len(edge_feats), " elem 0 shape", edge_feats[0].shape)
