import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.hpo_configs import RandHyperparamsConfig
from config import n_architectures
from resources.data import upload_torch_tensor, upload_dataset
import time

if __name__ == "__main__":
    """
    Benchmark. Time to train 15 architectures, 50 epochs each.

    on single A10G GPU: 
    """

    start_time = time.time()
    config = RandHyperparamsConfig(n_epochs_range=[1, 2])

    result = train_random_cnns_hyperparams("data/hpo", n_architectures=n_architectures, random_hyperparams_config = config)
    print(f"Time taken: {time.time() - start_time:.2f} seconds.")

    upload_dataset(*result, parent_dir="test-hpo")

