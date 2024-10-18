import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import n_architectures, n_epochs_range
from preprocessing.generate_data import train_random_cnns_random_hyperparams
from preprocessing.preprocessing_types import (
    RandHyperparamsConfig,
    RandCNNConfig,
    Hyperparameters
)
from resources import AzureFileClient, LocalFileClient, HPOExperimentClient
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', "--filesystem", type=str, choices=["local", "azure"], default="local",
    )
    parser.add_argument("-d", "--results_dir", type=str, default="./data/fixed_hp_data/")

    args = parser.parse_args()

    if args.filesystem == "local":
        file_client = LocalFileClient(args.results_dir)
    elif args.filesystem == "azure":
        file_client = AzureFileClient(args.results_dir)

    dataset_client = HPOExperimentClient(file_client=file_client)

    hps = Hyperparameters(lr=0.001, batch_size=256, n_epochs=30)

    hps_list = [hps for _ in range(n_architectures)]


    train_random_cnns_random_hyperparams(
        n_architectures=n_architectures,
        random_cnn_config=RandCNNConfig(),
        hyperparams_list=hps_list,
        save_result_callback=dataset_client.save_model_result,
    )
