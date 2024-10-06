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
from argparse import ArgumentParser



torch.manual_seed(0)
if __name__ == "__main__":
    """
    Train gmn on random CNNs trained with random hyperparameters
    """

    args = ArgumentParser()
    args.add_argument("--results_dir", type=str, default="data/hpo_result", help="Directory to save results")
    args.add_argument("--node_feat_dim", type=int, default=3)
    args.add_argument("--edge_feat_dim", type=int, default=6)
    args.add_argument("--node_hidden_dim", type=int, default=16)
    args.add_argument("--edge_hidden_dim", type=int, default=16)
    args.add_argument("--hidden_dim", type=int, default=8)
    args.add_argument("--batch_size", type=int, default=2)
    args.add_argument("--valid_size", type=float, default=0.1)
    args.add_argument("--test_size", type=float, default=0.1)
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--lr", type=float, default=0.01)
    args = args.parse_args()
    client = HPOExperimentClient(LocalFileClient("data/cnn_hpo"))
    dataset = client.read_dataset()

    features, labels = client.read_dataset()

    hpo_gmn = train_hpo(args, features, labels)
