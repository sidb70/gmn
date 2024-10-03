import torch
import torch.nn as nn
from .preprocessing_types import RandCNNConfig, RandMLPConfig


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)



def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def generate_random_cnn(config=RandCNNConfig()) -> nn.Sequential:
    """
    Generates a CNN classifier with varying convolutional and linear layers, as
    well as varying hidden units in each layer. Assumes input and kernels are 2d and square.

    Args:
    config: RandCNNConfig, the configuration for the random CNN model

    Returns:
    - model: nn.Sequential, the randomly generated CNN model
    """

    in_channels = config.in_channels
    in_dim = config.in_dim
    n_classes = config.n_classes
    kernel_size = config.kernel_size
    n_conv_layers_range = config.n_conv_layers_range
    n_fc_layers_range = config.n_fc_layers_range
    log_hidden_channels_range = config.log_hidden_channels_range
    log_hidden_fc_units_range = config.log_hidden_fc_units_range
    use_avg_pool_prob = config.use_avg_pool_prob
    pool_after_conv = config.pool_after_conv

    layers = []  # list of nn.Module

    # 1: conv layers
    conv_layer_number = 0
    n_conv_layers = torch.randint(*n_conv_layers_range, (1,)).item()

    while conv_layer_number < n_conv_layers:

        if conv_layer_number != 0:
            # for all layers except the first, the in dim is the out dim of the previous layer,
            in_dim = conv_out_dim
            # and the in channels is the out channels of the previous layer
            in_channels = out_channels

            # check if kernel size is too large for the current in_dim
            if in_dim < kernel_size:
                break

        # For each conv layer, randomly determine the number of hidden channels
        out_channels = 2 ** torch.randint(*log_hidden_channels_range, (1,)).item()
        padding = 0

        # calculate the output shape of the conv layer
        conv_out_dim = in_dim - kernel_size + 1 + 2 * padding

        # add the conv layer, batch norm and activation
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        )
        if pool_after_conv:
            layers.append(nn.MaxPool2d(2))
            conv_out_dim = conv_out_dim // 2

        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())

        conv_layer_number += 1

    # 2: conv to linear layers transition
    if torch.rand(1).item() < use_avg_pool_prob:
        # replace last two dimensions with 1 because we apply global pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        # TODO: fix norm_to_graph() in model_arch_graph when use_avg_pool_prob is False
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
            fc_out_dim = 2 ** torch.randint(*log_hidden_fc_units_range, (1,)).item()
        else:
            fc_out_dim = n_classes

        # add the linear layer and activation
        layers.append(nn.Linear(fc_in_dim, fc_out_dim))
        layers.append(nn.ReLU())
        fc_in_dim = fc_out_dim
        linear_layer_number += 1

    # combine layers into a sequential
    seq = nn.Sequential(*layers)
    seq.apply(init_weights)
    return seq


def generate_random_mlp(config=RandMLPConfig()) -> nn.Sequential:
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

    in_dim = config.in_dim
    out_dim = config.out_dim
    n_layers_range = config.n_layers_range
    log_hidden_units_range = config.log_hidden_units_range

    layers = []  # list of tuples (layer: nn.Module, out_shape: torch.Size)

    linear_layer_number = 0
    n_layers = torch.randint(*n_layers_range, (1,)).item()

    while linear_layer_number < n_layers:

        in_dim = in_dim if linear_layer_number == 0 else layers[-1][1][0]

        if linear_layer_number < n_layers - 1:
            # for all layers except the last, randomly determine the number of hidden units
            layer_out_dim = 2 ** torch.randint(*log_hidden_units_range, (1,)).item()
        else:
            layer_out_dim = out_dim

        # add the linear layer and activation
        layers.append([nn.Linear(in_dim, layer_out_dim), torch.Size([layer_out_dim])])
        layers.append([nn.ReLU(), torch.Size([layer_out_dim])])
        linear_layer_number += 1

    # combine layers into a sequential
    return nn.Sequential(*[layer[0] for layer in layers])
