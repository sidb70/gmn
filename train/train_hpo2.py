import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import List
import torch
import torch.nn as nn
from models.mpnn_models import HPOMPNN
import numpy as np
from preprocessing.preprocessing_types import NetFeatures


def train_hpo_mpnn(features: List[NetFeatures], labels):
    """
    Train a MPNN to predict labels from network features and hyperparams

    Args:
    - features: NetFeatures
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    hidden_dim = 64
    lr = 0.001
    n_epochs = 10
    # last element of each feature vector is the hpo vector
    hpo_dim = len(features[0][-1])  # 4

    model = HPOMPNN(hidden_dim, hpo_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):

        for i in range(0, len(features), batch_size):
            outs = []
            for j in range(i, min(i + batch_size, len(features))):
                node_feat, edge_index, edge_feat, hpo_vec = features[j]
                node_feat, edge_index, edge_feat, hpo_vec = (
                    torch.tensor(node_feat, dtype=torch.float32).to(DEVICE),
                    torch.tensor(edge_index).to(DEVICE),
                    torch.tensor(edge_feat).to(DEVICE),
                    torch.tensor(hpo_vec).to(DEVICE),
                )
                out = model.forward(node_feat, edge_index, edge_feat, hpo_vec)
                outs.append(out)

            outs = torch.cat(outs, dim=1).squeeze(0).to(DEVICE)
            y = torch.tensor(labels[i : i + batch_size]).to(DEVICE)
            loss = criterion(outs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Loss: ", loss)

    return model
