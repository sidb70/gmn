import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import shutil
from preprocessing.generate_data import train_random_cnns_random_hyperparams
from train.utils import split
from train.train_hpo import train_hpo
from resources import LocalFileClient, HPOExperimentClient, LocalFileClient
from config import local_hpo_data_dir


def load_data(results_dir="cnn_hpo"):

    experiment = HPOExperimentClient(
        LocalFileClient(os.path.join(local_hpo_data_dir, results_dir))
    )

    features, labels = experiment.read_dataset()

    print(len(features), len(labels))

    valid_size = 0.1
    test_size = 0.1
    train_set, valid_set, test_set = split(features, labels, test_size, valid_size)

    return train_set, valid_set, test_set


if __name__ == "__main__":
    """
    Train gmn on random CNNs trained with random hyperparameters
    """

    torch.manual_seed(0)

    results_dir = "data/hpo_result"

    train_set, valid_set, test_set = load_data(results_dir)
    print(len(train_set), len(valid_set), len(test_set))
    exit(0)
    client = HPOExperimentClient(LocalFileClient("data/cnn_hpo"))
    dataset = client.read_dataset()

    features, val_losses = client.read_dataset()

    valid_size = 0.1
    test_size = 0.1
    train_set, valid_set, test_set = split(features, val_losses, test_size, valid_size)

    feats_train, labels_train = train_set
    feats_valid, labels_valid = valid_set
    feats_test, labels_test = test_set

    hpo_gmn = train_hpo_mpnn(feats_train, labels_train)
