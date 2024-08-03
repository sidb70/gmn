import torch
import unittest
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.generate_data import train_random_cnns
from preprocessing.data_loader import get_dataset
from preprocessing.generate_nns import generate_random_cnn, generate_random_mlp


# Assuming this file is in tests/


class TestGenerateData(unittest.TestCase):

    # @unittest.skip("skip")
    def test_create_and_load_cnn_graph(self):

        n_architectures = 2

        train_random_cnns(
            n_architectures=n_architectures, train_size=100, batch_size=3, directory='data/test/cnn',
            replace_if_existing=True, log_hidden_channels_range=(2, 4), log_hidden_fc_units_range=(2, 4),
        )

        feats, labels = torch.load(
            'data/test/cnn/features.pt'), torch.load('data/test/cnn/accuracies.pt')

        self.assertEqual(feats[0][0].shape[1], 3)
        self.assertEqual(feats[0][1].shape[1], feats[0][2].shape[0])
        self.assertEqual(len(labels), n_architectures)


if __name__ == '__main__':
    unittest.main()
