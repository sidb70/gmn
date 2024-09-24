import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import shutil
from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.types import RandHyperparamsConfig, RandCNNConfig
from train.utils import split
from train.train_hpo2 import train_hpo_mpnn


def load_data(results_dir):
    os.makedirs(results_dir, exist_ok=True)

    features = torch.load(os.path.join(results_dir, "features.pt"))
    accuracies = torch.load(os.path.join(results_dir, "accuracies.pt"))

    print(len(features), len(accuracies))
    print(type(features[0][0]))
    print(len(features[0]))

    print(features[0][0].shape, features[0][1].shape, features[0][2].shape)
    print(accuracies[0])

    valid_size = 0.1
    test_size = 0.1
    train_set, valid_set, test_set = split(features, accuracies, test_size, valid_size)

    return train_set, valid_set, test_set


if __name__ == "__main__":
    """
    Train gmn on random CNNs trained with random hyperparameters
    """

    torch.manual_seed(0)

    results_dir = "data/hpo"

    generate_new_data = False

    if generate_new_data:
        shutil.rmtree(results_dir)

    features = torch.load(os.path.join(results_dir, "features.pt"))
    # list of tuples (node_feats, edge_indices, edge_feats, hpo_vec)
    accuracies = torch.load(os.path.join(results_dir, "accuracies.pt"))
    # list of accuracies

    print(len(features), len(accuracies))
    print(type(features[0][0]))
    print(len(features[0]))

    print(features[0][0].shape, features[0][1].shape, features[0][2].shape)
    print(accuracies[0])

    valid_size = 0.1
    test_size = 0.1
    train_set, valid_set, test_set = split(features, accuracies, test_size, valid_size)

    feats_train, labels_train = train_set
    feats_valid, labels_valid = valid_set
    feats_test, labels_test = test_set

    hpo_gmn = train_hpo_mpnn(feats_train, labels_train)
