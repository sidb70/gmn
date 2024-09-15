import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.hpo_configs import RandHyperparamsConfig
from config import n_architectures
import time

if __name__ == "__main__":
    start_time = time.time()
    config = RandHyperparamsConfig(n_epochs_range=[50, 150])
    train_random_cnns_hyperparams("data/hpo", n_architectures=n_architectures)
    print(f"Time taken: {time.time() - start_time:.2f} seconds.")

    # os.makedirs(results_dir, exist_ok=True)

    # n_architectures = 15

    # train_random_cnns_hyperparams(
    #     "data/hpo",
    #     random_cnn_config=RandCNNConfig(
    #         n_classes=10,
    #         n_conv_layers_range=(2, 3),
    #         n_fc_layers_range=(2, 3),
    #         log_hidden_channels_range=(6, 7),
    #         log_hidden_fc_units_range=(6, 7),
    #         use_avg_pool_prob=True,
    #     ),
    #     random_hyperparams_config=RandHyperparamsConfig(
    #         n_epochs_range=(2, 3),
    #         log_batch_size_range=(2, 5),
    #     ),
    #     n_architectures=n_architectures,
    # )
