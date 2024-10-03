import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.preprocessing_types import RandHyperparamsConfig, RandCNNConfig
from preprocessing.preprocessing_types import HPOFeatures
from config import n_architectures, n_epochs_range
from resources.file_clients import AzureFileClient
from azure.core.exceptions import ResourceNotFoundError
import time
import torch


def train_save_to_azure(client: AzureFileClient):

    try:
        features, accuracies = client.fetch_dataset()
    except ResourceNotFoundError:
        features, accuracies = [], []

    def save_to_azure_callback(feature: HPOFeatures, accuracy: float, model_idx: int):
        features.append(feature)
        accuracies.append(accuracy)
        print(model_idx, n_architectures)
        if (model_idx + 1) % 100 == 0:
            client.upload_dataset(features, accuracies)

    random_cnn_config = RandCNNConfig()
    random_hyperparams_config = RandHyperparamsConfig(n_epochs_range=n_epochs_range)

    train_random_cnns_hyperparams(
        random_cnn_config=random_cnn_config,
        random_hyperparams_config=random_hyperparams_config,
        n_architectures=n_architectures,
        save_data_callback=save_to_azure_callback,
    )

    client.upload_dataset(features, accuracies)


def train_save_locally(
    results_dir="data/cnn_hpo",
    random_cnn_config = RandCNNConfig(),
    random_hyperparams_config = RandHyperparamsConfig(n_epochs_range=n_epochs_range)
):

    os.makedirs(results_dir, exist_ok=True)

    """
    callback every model trained:
    - save current epoch's
    """

    def save_locally_callback(model_feats, hpo_vec, train_losses, val_losses, accuracy, model_id):
        model_dir = os.path.join(results_dir, f"{model_id}")
        if os.path.exists(model_dir):
            os.rmdir(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        torch.save(model_feats, os.path.join(model_dir, "model_features.pt")) 
        results = {
            "hyperparameters": hpo_vec,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "accuracy": accuracy,
        }
        with open(os.path.join(model_dir, "results.json"), "w") as f:
            json.dump(results, f)
        print("Saved model {} to {}".format(model_id, model_dir))


    train_random_cnns_hyperparams(
        random_cnn_config=random_cnn_config,
        random_hyperparams_config=random_hyperparams_config,
        n_architectures=n_architectures,
        save_data_callback=save_locally_callback,
    )



if __name__ == "__main__":
    """
    Benchmark. Time to train 15 architectures, 50 epochs each.

    on single A10G GPU:
    """

    # client = AzureDatasetClient()
    # train_save_to_azure(client)
    train_save_locally()
    exit(0)
