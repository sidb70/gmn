import torch
import torch.nn as nn


def generate_random_cnn(in_dim=32, in_channels=3, out_dim=10, avg_n_conv_layers=3, avg_n_fc_layers=2, n_log_out_channels=4) -> nn.Module:
  """
  creates a cnn with a random number of convolutional layers followed by random linear layers
  assumes input to the cnn is (batch size, in_channels, in_dim, in_dim) shape

  Args:
  - in_dim: int, the height and width of the input image
  - in_channels: int, the number of channels in the input image
  - out_dim: int, the number of classes
  - n_log_out_channels (optional): int, the log of the minimum number of output channels for the convolutional layers

  Returns:
  - model: nn.Module, the randomly generated CNN model

  """

  layers = []  # list of tuples (layer, out_shape)

  while True:
    if len(layers) == 0:
      out_channels = 2**torch.randint(n_log_out_channels, n_log_out_channels+4, (1,)).item()
      layers.append((nn.Conv2d(in_channels, out_channels, 3),
                    (out_channels, in_dim-2, in_dim-2)))

    elif isinstance(layers[-1][0], nn.Conv2d):
      # batch norm layer
      n_features = layers[-1][0].out_channels
      layers.append((nn.BatchNorm2d(n_features), layers[-1][1]))
      layers.append((nn.ReLU(), layers[-1][1]))

      # randomly either add another conv layer or switch to linear layers
      if torch.rand(1).item() > (1/avg_n_conv_layers) and \
              layers[-1][1][1] > 3 and layers[-1][1][2] > 3:  # check if out shape is smaller than the kernel size
        out_channels = 2**torch.randint(n_log_out_channels, n_log_out_channels+4, (1,)).item()
        layers.append((nn.Conv2d(layers[-1][1][0], out_channels, 3),
                      (out_channels, layers[-1][1][1]-2, layers[-1][1][2]-2)))
      else:
        # begin linear layers
        layers.append(
            (nn.Flatten(), (layers[-1][1][0]*layers[-1][1][1]*layers[-1][1][2],)))
        continue

    elif isinstance(layers[-1][0], nn.Flatten) or isinstance(layers[-1][0], nn.ReLU):
      # linear layer
      if torch.rand(1).item() > 1/avg_n_fc_layers:
        out_features = 2**torch.randint(2, 6, (1,)).item()
        layers.append(
            (nn.Linear(layers[-1][1][0], out_features), (out_features,)))
        layers.append((nn.ReLU(), layers[-1][1]))
      else:
        layers.append((nn.Linear(layers[-1][1][0], out_dim), (out_dim,)))
        break

  return nn.Sequential(*[layer[0] for layer in layers])


def generate_random_mlp(in_dim=32, out_dim=10, avg_n_layers=4):
  """
  creates a sequential model with random number of linear layers and random number of units in each layer

  Args:
  - in_dim: int, the number of input features
  - out_dim: int, the number of output features

  Returns:
  - model: nn.Module, the randomly generated model
  """

  layers = []  # list of tuples (layer, out_shape)

  while True:
    if len(layers) == 0:
      # add input layer
      out_features = 2**torch.randint(4, 8, (1,)).item()
      layers.append((nn.Linear(in_dim, out_features), (out_features,)))
      layers.append((nn.ReLU(), layers[-1][1]))

    elif isinstance(layers[-1][0], nn.ReLU):

      # randomly either add another linear layer or add final layer
      if torch.rand(1).item() < 1/avg_n_layers:
        out_features = 2**torch.randint(4, 10, (1,)).item()
        layers.append(
            (nn.Linear(layers[-1][1][0], out_features), (out_features,)))
        layers.append((nn.ReLU(), layers[-1][1]))
      else:
        layers.append((nn.Linear(layers[-1][1][0], out_dim), (out_dim,)))
        break

  return nn.Sequential(*[layer[0] for layer in layers])
