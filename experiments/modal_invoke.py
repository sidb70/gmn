import modal

# modal.Function.lookup("hpo", "generate_data").remote()
modal.Function.lookup("hpo", "train_hpo_mpnn").remote()
