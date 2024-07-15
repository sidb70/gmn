from param_graph.preprocessing.generate_data import generate_data

if __name__ == '__main__':
  generate_data(
    n_architectures=1,
    train_proportion=.8,
    n_epochs=50,
  )
