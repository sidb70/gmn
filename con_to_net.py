from graph_types import (NetworkLayer, Node, Edge, NodeType, EdgeType, LayerType,
                         NodeFeatures, EdgeFeatures)
from factory import LayerFactory
import torch.nn as nn
from networkx import Graph

def conv_to_net (conv: nn.Conv2d) -> list[NetworkLayer]:
    '''
    Convert a PyTorch Conv2d layer to a list of NetworkLayer objects

    Args:
    - conv (nn.Conv2d): PyTorch Conv2d layer

    Returns:
    - NetworkLayer: NetworkLayer object
    '''
    layer_factory = LayerFactory()
    layers = []
    layer = layer_factory.create_layer(conv, layer_num=0, start_node_id=0)
    return layer