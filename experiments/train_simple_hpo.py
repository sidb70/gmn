import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train.train_simple_hpo import train_simple_hpo
import torch


if __name__ == "__main__":

    hpo_data_dir = "data/hpo"

    feats, labels = \
      torch.load(os.path.join(hpo_data_dir, "features.pt")), \
      torch.load(os.path.join(hpo_data_dir, "accuracies.pt"))

    train_simple_hpo(feats, labels)


    
