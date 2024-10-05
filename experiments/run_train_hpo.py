import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import shutil
from train.utils import split
from train.train_hpo import train_hpo
from resources.dataset_clients import HPOExperimentClient, LocalFileClient


def load_data(results_dir):
    os.makedirs(results_dir, exist_ok=True)

    client = HPOExperimentClient(LocalFileClient('/mnt/home/bhatta70/Documents/gmn/gmn/data/cnn_hpo'))

    dataset = client.read_dataset()
    features, labels = dataset


    print(len(features), len(labels))


    # features = torch.load(os.path.join(results_dir, "features.pt"))
    # accuracies = torch.load(os.path.join(results_dir, "accuracies.pt"))

    # print(len(features), len(accuracies))
    # print(type(features[0][0]))
    # print(len(features[0]))

    # print(features[0][0].shape, features[0][1].shape, features[0][2].shape)
    # print(accuracies[0])

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

    feats_train, labels_train = train_set
    feats_valid, labels_valid = valid_set
    feats_test, labels_test = test_set

    hpo_gmn = train_hpo_mpnn(feats_train, labels_train)
