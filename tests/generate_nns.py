import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import unittest
from preprocessing.generate_nns import (
    generate_random_cnn,
    RandCNNConfig,
    generate_random_mlp,
    RandMLPConfig,
)


class TestGenerateNNs(unittest.TestCase):

    # @unittest.skip("skip")
    def test_generate_random_cnn(self):
        torch.manual_seed(0)

        cnn = generate_random_cnn()
        sample_input = torch.randn(1000, 3, 32, 32)
        output = cnn(sample_input)
        self.assertEqual(output.shape, torch.Size([1000, 10]))

        cnn = generate_random_cnn(
            RandCNNConfig(
                in_dim=64, in_channels=13, n_classes=8, n_conv_layers_range=(10, 20),
                use_avg_pool_prob=False
            )
        )
        sample_input = torch.randn(10, 13, 64, 64)
        output = cnn(sample_input)
        self.assertEqual(output.shape, torch.Size([10, 8]))

    # @unittest.skip("skip")
    def test_generate_random_mlp(self):
        torch.manual_seed(0)

        mlp = generate_random_mlp(RandMLPConfig(in_dim=32, out_dim=10))
        sample_input = torch.randn(1000, 32)
        output = mlp(sample_input)
        self.assertEqual(output.shape, torch.Size([1000, 10]))


if __name__ == "__main__":
    unittest.main()
