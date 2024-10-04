from dataclasses import dataclass
from typing import Tuple, List, Union
import torch
import numpy as np


HPOvec = List[Union[int, float]]


@dataclass(frozen=True)
class Hyperparameters:
    """
    Dataclass for hyperparameters to use for training HPO models.
    """

    batch_size: int = 3
    lr: float = 0.01
    n_epochs: int = 1
    momentum: float = 0.5

    def to_vec(self) -> HPOvec:
        return [self.batch_size, self.lr, self.n_epochs, self.momentum]

    def __str__(self):
        return (
            f"batch_size: {self.batch_size}, lr: {self.lr:.4f}, "
            f"n_epochs: {self.n_epochs}, momentum: {self.momentum:.4f}"
        )


@dataclass(frozen=True)
class RandHyperparamsConfig:
    """
    Dataclass for random hyperparameters configuration.
    """

    # default values for the hyperparameters
    log_batch_size_range: Tuple[int, int] = (5, 11)
    lr_range: Tuple[float, float] = (0.0001, 0.07)
    n_epochs_range: Tuple[int, int] = (50, 150)
    momentum_range: Tuple[float, float] = (0.1, 0.9)

    def sample(self) -> Hyperparameters:
        """
        Samples hyperparameters from the configuration.
        """
        return Hyperparameters(
            batch_size=2 ** np.random.randint(*self.log_batch_size_range),
            lr=np.random.uniform(*self.lr_range),
            n_epochs=np.random.randint(*self.n_epochs_range),
            momentum=np.random.uniform(*self.momentum_range),
        )


@dataclass(frozen=True)
class RandCNNConfig:
    in_channels: int = 3
    in_dim: int = 32
    # the dimension of the input image (assuming square)
    n_classes: int = 10
    kernel_size: int = 3
    n_conv_layers_range: Tuple[int, int] = (2, 4)
    n_fc_layers_range: Tuple[int, int] = (2, 4)
    log_hidden_channels_range: Tuple[int, int] = (4, 8)
    # n hidden channels = 2^sampled value
    log_hidden_fc_units_range: Tuple[int, int] = (4, 8)
    # n hidden fc units = 2^sampled value
    use_avg_pool_prob: float = 1.0
    pool_after_conv: bool = False


@dataclass(frozen=True)
class RandMLPConfig:
    in_dim: int = 32
    out_dim: int = 10
    n_layers_range: Tuple[int, int] = (2, 4)
    log_hidden_units_range: Tuple[int, int] = (4, 8)
    # n hidden units = 2^sampled value


@dataclass(frozen=True)
class NetFeatures:
    """
    Node and edge feats for a neural net's param graph
    Is based on the return type of seq_to_feats
    """

    node_feats: torch.Tensor  # shape (n_nodes, 3)
    edge_indices: torch.Tensor  # shape (2, n_edges)
    edge_feats: torch.Tensor  # shape (n_edges, 6)


# outdated
@dataclass(frozen=True)
class HPOFeatures(NetFeatures):
    """
    Includes the hyperparameters used to train it.
    """

    hpo_vec: HPOvec


# outdated
HPODataset = Tuple[List[HPOFeatures], List[float]]


@dataclass
class TrainedNNResult:
    """
    This class represents the result of training one
    neural net during preprocessing for HPO.
    """

    model_id: int
    # Unique identifier for the model architecture in this training run
    epoch_feats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    # Features of the model after each epoch
    train_losses: List[float]
    # Training losses after each epoch
    val_losses: List[float]
    # Validation losses after each epoch
    final_accuracy: float
    # Accuracy of the model on the test set
    hpo_vec: HPOvec
    # Hyperparameters vector used for the model
    device: torch.device
    # Device used for training the model


__all__ = [
    "Hyperparameters",
    "RandHyperparamsConfig",
    "RandCNNConfig",
    "RandMLPConfig",
    "NetFeatures",
    "HPOFeatures",
    "HPOvec",
    "TrainedNNResult",
    "HPODataset",
]
