import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


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


def train_simple_hpo(features, accuracies):

    simple_features = []
    for i in range(len(features)):
        n_params = len(features[i][1][0])
        hpo_vec = features[i][-1]
        simple_features.append([n_params] + hpo_vec)

    # normalize features
    scaler = StandardScaler()
    simple_features = scaler.fit_transform(simple_features)

    model = AccuracyPredictor(input_dim=len(simple_features[0]), hidden_dim=64)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
          
          for i in range(len(simple_features)):
              x = torch.tensor(simple_features[i], dtype=torch.float32)
              y = torch.tensor(accuracies[i])
  
              optimizer.zero_grad()
              output = model(x)
              loss = criterion(output, y)
              loss.backward()
              optimizer.step()
  
              print(f"Epoch {epoch}, Loss: {loss}")

    return model
