from preprocessing.generate_data import train_random_cnns, train_random_cnns_hyperparams

if __name__ == '__main__':
    # train_random_cnns(n_architectures=5, replace_if_existing=True)
    train_random_cnns_hyperparams(n_architectures=50000,  replace_if_existing=True)
