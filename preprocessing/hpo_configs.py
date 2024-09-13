from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Hyperparameters:
    """
    Dataclass for hyperparameters to use for training HPO models.
    """

    batch_size: int = 3
    lr: float = 0.01
    n_epochs: int = 1
    momentum: float = 0.5

    def to_vec(self):
        return [self.batch_size, self.lr, self.n_epochs, self.momentum]

    def __str__(self):
        return f"batch_size: {self.batch_size}, lr: {self.lr}, n_epochs: {self.n_epochs}, momentum: {self.momentum}"


@dataclass
class RandHyperparamsConfig:
    """
    Dataclass for random hyperparameters configuration.
    """

    # default values for the hyperparameters
    log_batch_size_range: Tuple[int, int] = (1, 10)
    lr_range: Tuple[float, float] = (0.0001, 0.1)
    n_epochs_range: Tuple[int, int] = (50, 150)
    momentum_range: Tuple[float, float] = (0.1, 0.9)

    def sample(self) -> Hyperparameters:
        """
        Samples n hyperparameters from the configuration.
        """
        return Hyperparameters(
            batch_size=np.random.randint(*self.log_batch_size_range),
            lr=np.random.uniform(*self.lr_range),
            n_epochs=np.random.randint(*self.n_epochs_range),
            momentum=np.random.uniform(*self.momentum_range),
        )


@dataclass
class RandCNNConfig:
    in_channels: int = 3
    in_dim: int = 32 # the dimension of the input image (assuming square)
    n_classes: int = 10
    kernel_size: int = 3
    n_conv_layers_range: Tuple[int, int] = (2, 4)
    n_fc_layers_range: Tuple[int, int] = (2, 4)
    log_hidden_channels_range: Tuple[int, int] = (4, 8) # n hidden channels = 2^sampled value
    log_hidden_fc_units_range: Tuple[int, int] = (4, 8) # n hidden fc units = 2^sampled value
    use_avg_pool_prob: float = 1.0


@dataclass
class RandMLPConfig:
    in_dim: int = 32
    out_dim: int = 10
    n_layers_range: Tuple[int, int] = (2, 4)
    log_hidden_units_range: Tuple[int, int] = (4, 8) # n hidden units = 2^sampled value
