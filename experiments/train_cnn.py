import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.generate_data import train_cnns_cfira10, RandCNNConfig, Hyperparameters
from preprocessing.generate_nns import generate_random_cnn
import torch


if __name__ == '__main__':
  torch.manual_seed(1)

  random_cnn_config = RandCNNConfig(
      n_classes=10,
      kernel_size=5,
      n_conv_layers_range=(2, 3),
      n_fc_layers_range=(3, 4),
      log_hidden_channels_range=(4, 5),
      log_hidden_fc_units_range=(6, 8),
      use_avg_pool=False,
  )

  hyperparams = Hyperparameters(
      batch_size=4,
      lr=0.01,
      n_epochs=4,
      momentum=0.5,
  )


  save_dir = "data/hpo_temp"
  train_cnns_cfira10(
      results_dir=save_dir,
      random_cnn_config=random_cnn_config,
      n_architectures=1,
      replace_if_existing=True,
  )

  feats = torch.load(os.path.join(save_dir, "features.pt"))
  accuracies = torch.load(os.path.join(save_dir, "accuracies.pt"))


