import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.hpo_configs import RandHyperparamsConfig
from config import n_architectures
import time

if __name__ == "__main__":
    """
    Benchmark. Time to train 15 architectures, 50 epochs each.

    on single A10G GPU: 
    """

    start_time = time.time()
    config = RandHyperparamsConfig(n_epochs_range=[50, 150])
    train_random_cnns_hyperparams("data/hpo", n_architectures=n_architectures)
    print(f"Time taken: {time.time() - start_time:.2f} seconds.")
