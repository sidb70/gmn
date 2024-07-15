import unittest
import os
import sys

import torch

# Assuming this file is in tests/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.generate_data import train_random_cnns
from preprocessing.data_loader import get_dataset
from preprocessing.generate_nns import generate_random_cnn, generate_random_mlp


class TestPreprocessing(unittest.TestCase):

  @unittest.skip("skip")
  def generate_random_cnn(self):
    torch.manual_seed(0)

    cnn = generate_random_cnn()
    sample_input = torch.randn(1000, 3, 32, 32)
    output = cnn(sample_input)
    self.assertEqual(output.shape, torch.Size([1000, 10]))

    cnn = generate_random_cnn(
        in_dim=64, in_channels=13, n_classes=8, n_conv_layers_range=(10, 20))
    sample_input = torch.randn(10, 13, 64, 64)
    output = cnn(sample_input)
    self.assertEqual(output.shape, torch.Size([10, 8]))

  @unittest.skip("skip")
  def generate_random_mlp(self):
    torch.manual_seed(0)

    mlp = generate_random_mlp(in_dim=32, out_dim=10)
    sample_input = torch.randn(1000, 32)
    output = mlp(sample_input)
    self.assertEqual(output.shape, torch.Size([1000, 10]))

  def create_and_load_preprocessed_data(self):

    train_random_cnns(
      n_architectures=2, train_size=100, batch_size=3, directory='data/test/cnn', 
      log_hidden_channels_range=(2, 4), log_hidden_fc_units_range=(2, 4)
    )

    node_feats, edge_indices, edge_feats, labels = get_dataset(
        'data/test/cnn/features.pt', 'data/test/cnn/accuracies.pt')

    assert node_feats[0].shape[1] == 3
    assert edge_indices[0].shape[1] == edge_feats.shape[0]
    assert len(labels) == 2 and len(node_feats) == 2


if __name__ == '__main__':
  unittest.main()
