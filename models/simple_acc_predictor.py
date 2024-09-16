import torch
import torch.nn as nn


class AccuracyPredictor(nn.Module):
    """
    To benchmark the GMN,
    Simple neural network to predict accuracy from just hyperparameters and  
    """

    def __init__(self, input_dim, hidden_dim):
        super(AccuracyPredictor, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x