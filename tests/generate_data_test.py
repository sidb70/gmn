import torch
import unittest
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.generate_data import train_random_cnns, Hyperparameters
from preprocessing.data_loader import get_dataset
from preprocessing.generate_nns import generate_random_cnn, generate_random_mlp, RandCNNConfig


# Assuming this file is in tests/


class TestGenerateData(unittest.TestCase):

    # @unittest.skip("skip")
    def test_create_train_and_load_cnns(self):

        n_architectures = 3

        random_cnn_config = RandCNNConfig(
            log_hidden_channels_range=(2, 4), log_hidden_fc_units_range=(2, 4)
        )

        hyperparams = Hyperparameters(batch_size=3, lr=0.01, n_epochs=1, momentum=0.5)

        results_dir = 'data/test'
        os.makedirs(results_dir, exist_ok=True)
        train_random_cnns(
            results_dir=results_dir,
            hyperparams=hyperparams,
            n_architectures=n_architectures, 
            replace_if_existing=True, random_cnn_config=random_cnn_config
        )

        feats, labels = \
            torch.load(os.path.join(results_dir, 'features.pt')), torch.load(os.path.join(results_dir, 'accuracies.pt'))

        self.assertEqual(feats[0][0].shape[1], 3)
        self.assertEqual(feats[0][1].shape[1], feats[0][2].shape[0])
        self.assertEqual(len(labels), n_architectures)


if __name__ == '__main__':
    unittest.main()
