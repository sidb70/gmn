import modal

# modal.Function.lookup("hpo", "generate_data").remote()
modal.Function.lookup("hpo", "train_one_cnn").remote()
