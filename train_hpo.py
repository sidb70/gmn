import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import shutil
from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.generate_nns import RandCNNConfig
# from train.utils import split
from models.models import HPOMPNN
import numpy as np
from sklearn.model_selection import train_test_split


def train_hpo_gmn(features, accuracies):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gmn_batch_size = 4
    hidden_dim = 64
    hpo_dim = len(
        features[0][-1]
    )  # last element of each feature vector is the hpo vector

    model = HPOMPNN(hidden_dim)

    for i in range(0, len(features), gmn_batch_size):
        outs = []
        for j in range(i, min(i + gmn_batch_size, len(features))):
            node_feat, edge_index, edge_feat, hpo_vec = features[j]
            node_feat, edge_index, edge_feat, hpo_vec = (
                torch.tensor(node_feat).to(DEVICE),
                torch.tensor(edge_index).to(DEVICE),
                torch.tensor(edge_feat).to(DEVICE),
                torch.tensor(hpo_vec).to(DEVICE),
            )
            out = model.forward(node_feat, edge_index, edge_feat, hpo_vec)
            outs.append(out)




def split(features, accuracies, test_size, valid_size=None):
    """
    Split features and accuracies into train, test, and valid subsets.

    Args:
        features (list): List of feature vectors.
        accuracies (list): List of accuracies.
        test_size (float): Proportion of the data to include in the test set.
        valid_size (float, optional): Proportion of the data to include in the validation set. Defaults to None.

    Returns:
        tuple: Tuple containing train, test, and valid subsets of features and accuracies.
    """
    if valid_size is None:
        valid_size = 0.0

    # Shuffle the data
    features, accuracies = sklearn.utils.shuffle(features, accuracies)

    # Split the data into train, test, and valid subsets
    train_features, test_features, train_accuracies, test_accuracies = train_test_split(features, accuracies, test_size=test_size, random_state=0)
    train_features, valid_features, train_accuracies, valid_accuracies = train_test_split(train_features, train_accuracies, test_size=valid_size, random_state=0)

    return train_features, train_accuracies, test_features, test_accuracies, valid_features, valid_accuracies





if __name__ == "__main__":

    """
    Train gmn on random CNNs trained with random hyperparameters
    """

    torch.manual_seed(0)

    results_dir = "data/hpo"

    generate_new_data = False

    if generate_new_data:
        shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)
        random_cnn_config = RandCNNConfig(
            n_classes=10,
            n_conv_layers_range=(2, 3),
            n_fc_layers_range=(2, 3),
            log_hidden_channels_range=(6, 7),
            log_hidden_fc_units_range=(6, 7),
            use_avg_pool=True,
        )

        n_architectures = 15

        train_random_cnns_hyperparams(
            "data/hpo",
            random_cnn_config=random_cnn_config,
            n_architectures=n_architectures,
        )

    features = torch.load(os.path.join(results_dir, "features.pt")) 
    # list of tuples (node_feats, edge_indices, edge_feats, hpo_vec)
    accuracies = torch.load(os.path.join(results_dir, "accuracies.pt"))
    # list of accuracies


    print(type(features[0][0]))
    print(len(features[0]))

    print(features[0][0].shape, features[0][1].shape, features[0][2].shape)
    print(accuracies[0])

    # valid_size = 0.1
    # test_size = 0.1
    # train_set, valid_set, test_set = split(valid_size, test_size, features, accuracies)

    # feats_train, labels_train = train_set
    # feats_valid, labels_valid = valid_set
    # feats_test, labels_test = test_set

    # hpo_gmn = train_hpo_gmn(features, accuracies)

