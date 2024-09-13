from preprocessing.generate_data import train_random_cnns_hyperparams, RandHyperparamsConfig, RandCNNConfig

if __name__ == '__main__':
    train_random_cnns_hyperparams('data/hpo', n_architectures=2)
