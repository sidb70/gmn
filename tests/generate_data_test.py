import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import unittest
import shutil
from preprocessing.generate_data import train_random_cnns_random_hyperparams
from preprocessing.preprocessing_types import (
    Hyperparameters,
    RandCNNConfig,
    RandHyperparamsConfig,
    TrainedNNResult,
)

from resources import LocalFileClient, HPOExperimentClient


class TestGenerateData(unittest.TestCase):

    @unittest.skip("skip")
    def test_train_random_cnn(self):

        random_cnn_config = RandCNNConfig(
            n_classes=10,
            kernel_size=5,
            n_conv_layers_range=(2, 3),
            n_fc_layers_range=(3, 4),
            log_hidden_channels_range=(4, 5),
            log_hidden_fc_units_range=(6, 8),
        )

        rand_hyperparam_config = RandHyperparamsConfig(
            n_epochs_range=(4, 10),
        )

        file_client = LocalFileClient("data/test-train-cnns")

        model_ids = []

        def save_result_callback(result: TrainedNNResult):
            model_ids.append(result.model_id)
            file_client.obj_to_pt_file(
                result.epoch_feats[-1], f"model_{result.model_id}.pt"
            )

        train_random_cnns_random_hyperparams(
            n_architectures=5,
            random_cnn_config=random_cnn_config,
            random_hyperparams_config=rand_hyperparam_config,
            save_result_callback=save_result_callback,
        )

        # read the saved files for the first model
        feats = file_client.read_pt_file(f"model_{model_ids[0]}.pt")
        print(feats[0].shape, feats[1].shape, feats[2].shape)

        file_client.delete_directory()

    def test_train_random_cnns_random_hyperparams(self):

        data_client = HPOExperimentClient(LocalFileClient("data/test-train-cnns"))

        data_client.file_client.delete_directory()

        train_random_cnns_random_hyperparams(
            n_architectures=19,
            random_hyperparams_config=RandHyperparamsConfig(n_epochs_range=(4, 10)),
            save_result_callback=data_client.save_model_result,
        )

        # count the number of directories in the base directory
        dirs = os.listdir(data_client.file_client.base_dir)
        self.assertEqual(len(dirs), 19)


if __name__ == "__main__":
    unittest.main()
