import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import unittest
import shutil
from preprocessing.generate_data import (
    train_cnns_cfira10,
    Hyperparameters,
    RandCNNConfig
)


class TestGenerateData(unittest.TestCase):

    def test_train_random_cnn(self):

        torch.manual_seed(0)
        np.random.seed(0)

        random_cnn_config = RandCNNConfig(
            n_classes=10,
            kernel_size=5,
            n_conv_layers_range=(2, 3),
            n_fc_layers_range=(3, 4),
            log_hidden_channels_range=(4, 5),
            log_hidden_fc_units_range=(6, 8),
        )

        hyperparams = Hyperparameters(
            log_batch_size=4,
            lr=0.01,
            n_epochs=4,
            momentum=0.5,
        )

        random_cnn_config = RandCNNConfig(
            log_hidden_channels_range=(2, 4), 
            log_hidden_fc_units_range=(2, 4),
        )

        hyperparams = Hyperparameters(log_batch_size=3, lr=0.01, n_epochs=1, momentum=0.5)

        save_dir = "data/hpo_test"
        train_cnns_cfira10(
            n_architectures=1,
            results_dir=save_dir,
            random_cnn_config=random_cnn_config,
            hyperparams=hyperparams,
            replace_if_existing=True,
        )

        feats = torch.load(os.path.join(save_dir, "features.pt"))
        accuracies = torch.load(os.path.join(save_dir, "accuracies.pt"))

        shutil.rmtree(save_dir)


    @unittest.skip("skip")
    def test_train_cnns(self):
        """
        Generates random CNNs, train them, and save and load the resulting features and labels.
        """

        n_architectures = 3

        random_cnn_config = RandCNNConfig(
            log_hidden_channels_range=(2, 4), log_hidden_fc_units_range=(2, 4)
        )

        hyperparams = Hyperparameters(log_batch_size=3, lr=0.01, n_epochs=1, momentum=0.5)

        results_dir = "data/test"
        os.makedirs(results_dir, exist_ok=True)
        train_cnns_cfira10(
            results_dir=results_dir,
            hyperparams=hyperparams,
            n_architectures=n_architectures,
            replace_if_existing=True,
            random_cnn_config=random_cnn_config,
        )

        feats, labels = torch.load(
            os.path.join(results_dir, "features.pt")
        ), torch.load(os.path.join(results_dir, "accuracies.pt"))

        # delete the test results_dir
        shutil.rmtree(results_dir)

        self.assertEqual(feats[0][0].shape[1], 3)
        self.assertEqual(feats[0][1].shape[1], feats[0][2].shape[0])
        self.assertEqual(len(labels), n_architectures)



if __name__ == "__main__":
    unittest.main()
