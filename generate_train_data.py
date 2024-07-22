from preprocessing.generate_data import train_random_cnns

if __name__ == '__main__':
    train_random_cnns(n_architectures=10, train_size=1000, n_epochs=2, replace_if_existing=True)