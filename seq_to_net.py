from graph_types import (NetworkLayer, Node, Edge, NodeType, EdgeType, LayerType,
                         NodeFeatures, EdgeFeatures)
from factory import LayerFactory
import torch.nn as nn
from networkx import Graph


def seq_to_net(seq: nn.Sequential) -> list[NetworkLayer]:
    '''
    Convert a PyTorch sequential model to a list of NetworkLayer objects

    Args:
    - seq (nn.Sequential): PyTorch sequential model

    Returns:
    - list[NetworkLayer]: List of NetworkLayer objects
    '''
    layer_factory = LayerFactory()
    layers = []
    # create first layer
    first_layer = layer_factory.create_layer(seq[0], layer_num=0, start_node_id=0)
    layers.append(first_layer)
    print(first_layer)
    return
    for layer_num, module in enumerate(seq):
        node_id = max(layers[-1].get_node_ids()) + 1
        if isinstance(module, nn.BatchNorm1d):
            layer = NetworkLayer(layer_num, LayerType.NORM)
            node_features = NodeFeatures(layer_num=layer_num, 
                                         rel_index=0, 
                                         node_type=NodeType.NORM)
            node = Node(node_id, node_features)
            layer.add_node(node)



def main():
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.BatchNorm1d(10),
        nn.Linear(10, 10),
    )
    seq_to_net(model)

if __name__ == '__main__':
    main()


