from preprocessing.generate_data import train_random_cnns, train_random_cnns_hyperparams

if __name__ == '__main__':
    train_random_cnns(n_architectures=5, replace_if_existing=True, hpo_vec=[4, 0.01, 2, 0.9])
    # train_random_cnns_hyperparams(n_architectures=10,  replace_if_existing=True)
