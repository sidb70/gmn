from typing import Tuple
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.shape[0], -1)

def generate_random_cnn(
    in_channels: int = 3,
    in_dim: int = 32,
    n_classes: int = 10,
    n_conv_layers_range: Tuple[int, int] = (2, 4),
    n_fc_layers_range: Tuple[int, int] = (2, 4),
    log_hidden_channels_range: Tuple[int, int] = (4, 8),
    log_hidden_fc_units_range: Tuple[int, int] = (4, 8),
    use_max_pool: bool = True
) -> nn.Sequential:
  """
  Generates a CNN classifier with varying convolutional and linear layers, as 
  well as varying hidden units in each layer. Assumes input and kernels are 2d and square.

  Args:
  - in_channels: 
      int, the number of channels in the input image
  - in_dim:
      int, the dimension of the input image (assuming square)
  - n_classes: 
      int, the number of classes
  - n_conv_layers_range and n_fc_layers_range:
      tuple, number of conv layers and fc layers are uniformly distributed with these 
      ranges
  - log_hidden_channels_range and log_hidden_fc_units_range:
      tuple, the number of hidden channels after each conv layer or fc layer 
      is 2**x, where x is uniformly distributed in this range

  Returns:
  - model: nn.Sequential, the randomly generated CNN model
  """

  layers = [] # list of nn.Module

  # 1: conv layers
  conv_layer_number = 0
  n_conv_layers = torch.randint(*n_conv_layers_range, (1,)).item()

  while conv_layer_number < n_conv_layers:

    if conv_layer_number != 0:
      # for all layers except the first, the in dim is the out dim of the previous layer,
      in_dim = conv_out_dim
      # and the in channels is the out channels of the previous layer
      in_channels = out_channels

    # For each conv layer, randomly determine the number of hidden channels
    out_channels = 2**torch.randint(*log_hidden_channels_range, (1,)).item()
    kernel_size = 3
    padding = 0

    # calculate the output shape of the conv layer
    conv_out_dim = in_dim - kernel_size + 1 + 2*padding

    # add the conv layer, batch norm and activation
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())

    conv_layer_number += 1

  # 2: conv to linear layers transition
  if use_max_pool:
    # replace last two dimensions with 1 because we apply global pooling
    layers.append(nn.AdaptiveAvgPool2d((1,1)))
    flatten_out_shape = out_channels
  else:
    flatten_out_shape = out_channels * conv_out_dim * conv_out_dim

  # add flatten layer
  layers.append(Flatten())
  layers.append(nn.LayerNorm(flatten_out_shape))

  # 3: fc layers
  linear_layer_number = 0
  n_fc_layers = torch.randint(*n_fc_layers_range, (1,)).item()
  fc_in_dim = flatten_out_shape

  while linear_layer_number < n_fc_layers:

    if linear_layer_number < n_fc_layers - 1:
      # for all layers except the last, randomly determine the number of hidden units
      fc_out_dim = 2**torch.randint(*log_hidden_fc_units_range, (1,)).item()
    else:
      fc_out_dim = n_classes

    # add the linear layer and activation
    layers.append(nn.Linear(fc_in_dim, fc_out_dim))
    layers.append(nn.ReLU())
    fc_in_dim = fc_out_dim
    linear_layer_number += 1

  # combine layers into a sequential
  return nn.Sequential(*layers)


def generate_random_mlp(in_dim=32, out_dim=10, n_layers_range=(2,4), log_hidden_units_range=(4,8)):
  """
  Generates an MLP with varying linear layers and varying hidden units in each layer.

  Args:
  - in_dim: 
      int, the input dimension
  - out_dim:
      int, the output dimension
  - n_layers_range:
      tuple, number of layers is uniformly distributed with this range (inclusive)

  Returns:
  - model: nn.Sequential, the randomly generated MLP model
  """

  layers = [] # list of tuples (layer: nn.Module, out_shape: torch.Size)

  linear_layer_number = 0
  n_layers = torch.randint(*n_layers_range, (1,)).item()

  while linear_layer_number < n_layers:
    
    in_dim = in_dim if linear_layer_number == 0 else layers[-1][1][0]

    if linear_layer_number < n_layers - 1:
      # for all layers except the last, randomly determine the number of hidden units
      layer_out_dim = 2**torch.randint(*log_hidden_units_range, (1,)).item()
    else:
      layer_out_dim = out_dim

    # add the linear layer and activation
    layers.append([nn.Linear(in_dim, layer_out_dim), torch.Size([layer_out_dim])])
    layers.append([nn.ReLU(), torch.Size([layer_out_dim])])
    linear_layer_number += 1

  # combine layers into a sequential
  return nn.Sequential(*[layer[0] for layer in layers])
