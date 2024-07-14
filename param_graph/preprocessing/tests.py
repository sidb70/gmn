import unittest
import os
import sys

# Assuming this file is in param_graph/preprocessing/tests.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..\\..'))
from param_graph.preprocessing.generate_nns import generate_random_cnn, generate_random_mlp
import torch

class TestPreprocessing(unittest.TestCase):

  def test_generate_random_cnn(self):
    torch.manual_seed(0)

    cnn = generate_random_cnn()
    sample_input = torch.randn(1000, 3, 32, 32)
    output = cnn(sample_input)
    self.assertEqual(output.shape, torch.Size([1000, 10]))

    cnn = generate_random_cnn(in_dim=64, in_channels=13, n_classes=8, n_conv_layers_range=(10,20))
    sample_input = torch.randn(10, 13, 64, 64)
    output = cnn(sample_input)
    self.assertEqual(output.shape, torch.Size([10, 8]))

  def test_generate_random_mlp(self):
    torch.manual_seed(0)

    mlp = generate_random_mlp(in_dim=32, out_dim=10)
    sample_input = torch.randn(1000, 32)
    output = mlp(sample_input)
    self.assertEqual(output.shape, torch.Size([1000, 10]))


if __name__ == '__main__':
  unittest.main()
