import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import n_architectures, n_epochs_range
from preprocessing.generate_data import train_random_cnns_random_hyperparams
from preprocessing.preprocessing_types import (
    RandHyperparamsConfig,
    RandCNNConfig,
)
from resources.file_clients import AzureFileClient, LocalFileClient
from resources.dataset_clients import HPOExperimentClient


if __name__ == "__main__":

    # file_client = LocalFileClient("data/cnn_hpo")
    file_client = AzureFileClient("cnn_hpo")

    dataset_client = HPOExperimentClient(file_client=file_client)

    train_random_cnns_random_hyperparams(
        n_architectures=n_architectures,
        random_cnn_config=RandCNNConfig(),
        random_hyperparams_config=RandHyperparamsConfig(n_epochs_range=n_epochs_range),
        save_result_callback=dataset_client.save_model_result,
    )
