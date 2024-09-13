from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.hpo_configs import RandHyperparamsConfig, RandHyperparamsConfig

if __name__ == '__main__':
    config = RandHyperparamsConfig(n_epochs_range = [50,150])
    train_random_cnns_hyperparams('data/hpo', n_architectures=30000, random_hyperparams_config=config)
