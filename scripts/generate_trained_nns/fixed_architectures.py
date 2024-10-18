import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from scripts.util import parse_data_config_args
from preprocessing.generate_data import train_random_cnns_random_hyperparams
from preprocessing.preprocessing_types import RandCNNConfig, RandHyperparamsConfig
from resources import HPOExperimentClient
from config import n_architectures, n_epochs_range
import torch

# Generate random CNNs with fixed architectures, using random hyperparameters

if __name__ == "__main__":

    dataset_client = parse_data_config_args(default_directory="data/fixed-arch_cnn_hpo")

    train_random_cnns_random_hyperparams(
        save_result_callback=dataset_client.save_model_result,
        n_architectures=n_architectures,
        random_hyperparams_config=RandHyperparamsConfig(n_epochs_range=n_epochs_range),
        random_cnn_config=RandCNNConfig(
            n_conv_layers_range=(3, 4),
            n_fc_layers_range=(2, 3),
            log_hidden_channels_range=(5, 6),
            log_hidden_fc_units_range=(6, 7),
        ),
    )
