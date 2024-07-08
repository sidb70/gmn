from models.models import EdgeModel, NodeModel, BaseMPNN
from param_graph.seq_to_net import seq_to_net
import torch.nn as nn
import torch
import random
import numpy as np
from argparse import ArgumentParser
torch.manual_seed(0)
random.seed(0)

def main(args):
    node_feat_dim = args.node_feat_dim
    edge_feat_dim = args.edge_feat_dim
    node_hidden_dim = args.node_hidden_dim
    edge_hidden_dim = args.edge_hidden_dim
    model = BaseMPNN(node_feat_dim, edge_feat_dim, node_hidden_dim, edge_hidden_dim)
    

if __name__=='__main__':
    args = ArgumentParser()
    args.add_argument('--node_feat_dim', type=int, default=3)
    args.add_argument('--edge_feat_dim', type=int, default=6)
    args.add_argument('--node_hidden_dim', type=int, default=4)
    args.add_argument('--edge_hidden_dim', type=int, default=4)
    args = args.parse_args()
    main(args)

