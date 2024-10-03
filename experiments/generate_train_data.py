import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.preprocessing_types import RandHyperparamsConfig, RandCNNConfig
from preprocessing.preprocessing_types import HPOFeatures
from config import n_architectures, n_epochs_range
from resources.azure_files import AzureDatasetClient
from azure.core.exceptions import ResourceNotFoundError
import time
import torch


def train_save_to_azure(client: AzureDatasetClient):

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


def train_save_locally():

    results_dir = "data/cnn_hpo"

    os.makedirs(results_dir, exist_ok=True)

    features_path = os.path.join(results_dir, "features.pt")
    accuracies_path = os.path.join(results_dir, "accuracies.pt")
    if os.path.exists(os.path.join(results_dir, "features.pt")):
        features = torch.load(features_path)
    else:
        features = []
    if os.path.exists(accuracies_path):
        accuracies = torch.load(accuracies_path)
    else:
        accuracies = []

    print("Loaded", len(features), "features and", len(accuracies), "accuracies")

    def save_locally_callback(feature, accuracy):

        features.append(feature)
        accuracies.append(accuracy)
        if os.path.exists(features_path):
            os.remove(features_path)
        if os.path.exists(accuracies_path):
            os.remove(accuracies_path)

        torch.save(features, features_path)
        torch.save(accuracies, accuracies_path)
        print("saved to {}".format(results_dir))

    start_time = time.time()

    random_cnn_config = RandCNNConfig()
    random_hyperparams_config = RandHyperparamsConfig(n_epochs_range=[15, 35])
    train_random_cnns_hyperparams(
        "data/hpo",
        n_architectures=n_architectures,
        random_hyperparams_config=random_hyperparams_config,
        random_cnn_config=random_cnn_config,
        save_data_callback=save_locally_callback,
    )
    print(f"Time taken: {time.time() - start_time:.2f} seconds.")

    # upload_dataset(*result, parent_dir="test-hpo")


if __name__ == "__main__":
    """
    Benchmark. Time to train 15 architectures, 50 epochs each.

    on single A10G GPU:
    """

    client = AzureDatasetClient()
    train_save_to_azure(client)
    # train_save_locally()
    exit(0)

    start_time = time.time()

    random_cnn_config = RandCNNConfig()
    random_hyperparams_config = RandHyperparamsConfig(n_epochs_range=[15, 35])
    result = train_random_cnns_hyperparams(
        "data/hpo",
        n_architectures=n_architectures,
        random_hyperparams_config=random_hyperparams_config,
        random_cnn_config=random_cnn_config,
    )
    print(f"Time taken: {time.time() - start_time:.2f} seconds.")
