import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import n_architectures, n_epochs_range
from preprocessing.generate_data import train_random_cnns_random_hyperparams
from preprocessing.preprocessing_types import (
    RandHyperparamsConfig,
    RandCNNConfig,
)
from resources import AzureFileClient, LocalFileClient, HPOExperimentClient
import argparse


def get_dataset_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', "--filesystem", type=str, choices=["local", "azure"], default="local",
    )
    parser.add_argument("-d", "--results_dir", type=str, default="")

    args = parser.parse_args()

    if args.filesystem == "local":
        file_client = LocalFileClient(args.results_dir)
    elif args.filesystem == "azure":
        file_client = AzureFileClient(args.results_dir)

    return file_client


if __name__ == "__main__":

    file_client = get_dataset_arguments()

    dataset_client = HPOExperimentClient(file_client=file_client)

    train_random_cnns_random_hyperparams(
        n_architectures=n_architectures,
        random_cnn_config=RandCNNConfig(),
        random_hyperparams_config=RandHyperparamsConfig(n_epochs_range=n_epochs_range),
        save_result_callback=dataset_client.save_model_result,
    )
