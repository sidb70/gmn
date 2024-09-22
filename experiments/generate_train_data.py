import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.hpo_configs import RandHyperparamsConfig, RandCNNConfig
from config import n_architectures
from resources.azure_files import upload_torch_tensor, upload_dataset, load_pt_file
from azure.core.exceptions import ResourceNotFoundError
import time
import torch


def train_save_to_azure(base_dir='test-hpo', n_architectures=10):

    try:
        features = load_pt_file(os.path.join(base_dir, 'features.pt'))
        accuracies = load_pt_file(os.path.join(base_dir, 'accuracies.pt'))
    except ResourceNotFoundError:
        features, accuracies = [], []

    def save_to_azure_callback(feature, accuracy):
        
        features.append(feature)
        print('appending new feature to existing_features', len(features))
        accuracies.append(accuracy)

        upload_torch_tensor(features, os.path.join(base_dir, 'features.pt'))
        upload_torch_tensor(accuracies, os.path.join(base_dir, 'accuracies.pt'))

    random_cnn_config = RandCNNConfig()
    random_hyperparams_config = RandHyperparamsConfig(n_epochs_range=[1, 2])
    result = train_random_cnns_hyperparams( 
                        n_architectures=n_architectures,
                        random_hyperparams_config=random_hyperparams_config,
                        random_cnn_config = random_cnn_config,
                        save_data_callback=save_to_azure_callback)



def train_save_locally():

    results_dir = "data/hpo"

    os.makedirs(results_dir, exist_ok=True)

    features_path = os.path.join(results_dir, "features.pt")
    accuracies_path = os.path.join(results_dir, "accuracies.pt")
    if os.path.exists(os.path.join(results_dir, "features.pt")):
        features = torch.load(features_path)
    else:
        features = []
    if os.path.exists(accuracies_path):
        accuracies = torch.load(accuracies_path)
    else:
        accuracies = []

    def save_locally_callback(feature, accuracy):
    
        features.append(feature)
        accuracies.append(accuracy)
        if os.path.exists(features_path):
            os.remove(features_path)
        if os.path.exists(accuracies_path):
            os.remove(accuracies_path)

        torch.save(features, features_path)
        torch.save(accuracies, accuracies_path)

    start_time = time.time()

    random_cnn_config = RandCNNConfig()
    random_hyperparams_config = RandHyperparamsConfig(n_epochs_range=[1, 2])
    result = train_random_cnns_hyperparams("data/hpo", 
                        n_architectures=n_architectures, 
                        random_hyperparams_config=random_hyperparams_config,
                        random_cnn_config = random_cnn_config, 
                        save_data_callback=save_locally_callback)
    print(f"Time taken: {time.time() - start_time:.2f} seconds.")

    upload_dataset(*result, parent_dir="test-hpo")


if __name__ == "__main__":
    """
    Benchmark. Time to train 15 architectures, 50 epochs each.

    on single A10G GPU: 
    """

    train_save_locally()
    exit(0)


    start_time = time.time()

    random_cnn_config = RandCNNConfig()
    random_hyperparams_config = RandHyperparamsConfig(n_epochs_range=[1, 2])
    result = train_random_cnns_hyperparams("data/hpo", 
                        n_architectures=n_architectures, 
                        random_hyperparams_config=random_hyperparams_config,
                        random_cnn_config = random_cnn_config)
    print(f"Time taken: {time.time() - start_time:.2f} seconds.")

    upload_dataset(*result, parent_dir="test-hpo")
