import torch
import torch.nn as nn
from graph_types import NetworkLayer, Node, Edge, NodeType, EdgeType, LayerType
from networkx import Graph

def seq_to_net(seq: nn.Sequential) -> list[NetworkLayer]:
    layers = []
    for layer_num, module in enumerate(seq):
        if isinstance(module, nn.BatchNorm1d):
            layer = NetworkLayer(layer_num, LayerType.NORM)
            rel_index=0
            node_features = [layer_num, rel_index, NodeType.BNORM]


def main():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.BatchNorm1d(10),
        nn.Linear(10, 10),
    )
    seq_to_net(model)

if __name__ == '__main__':
    main()


