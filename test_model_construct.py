from param_graph.seq_to_net import seq_to_nx
from gmn_lim.model_arch_graph import seq_to_feats, tests
from preprocessing.generate_nns import generate_random_cnn, RandCNNConfig
import torch.nn as nn
import time

start=time.time()
model = generate_random_cnn(RandCNNConfig(
    in_dim=32,
    in_channels=3,
    n_classes=10,
    n_conv_layers_range=(10,15),
    n_fc_layers_range=(8,10),
    log_hidden_channels_range=(7,8),
    log_hidden_fc_units_range=(4,5),
))
print("Time to create random cnn:", round(time.time()-start,5))
print('total params', sum(p.numel() for p in model.parameters()))
start = time.time()
feats = seq_to_feats(model)
print([f.shape for f in feats])
print("Time to create feats:", round(time.time()-start,5))
# start = time.time()
# global_graph = seq_to_nx(model)
# print("NX param graph creation time", round(time.time()-start,5))
