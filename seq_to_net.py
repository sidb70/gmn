from graph_types import (NetworkLayer, Node, Edge, NodeType, EdgeType, LayerType,
                         NodeFeatures, EdgeFeatures)
from factory import LayerFactory
import torch.nn as nn
from networkx import MultiGraph


def seq_to_net(seq: nn.Sequential) -> MultiGraph:
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
    # print(first_layer)
    layer_num=1
    for module in seq:
        node_id = max(layers[-1].get_node_ids()) + 1
        layer = layer_factory.create_layer(module, layer_num=layer_num, start_node_id=node_id, prev_layer=layers[-1])
        layers.append(layer)
        layer_num += 1
    # for layer in layers:
    #     print("Layer: ", layer, "Nodes", layer.get_node_ids())
    global_graph = MultiGraph()
    for layer in layers:
        for node in layer.get_nodes():
            global_graph.add_node(node.node_id, node_obj=node)
        for edge in layer.get_edges():
            global_graph.add_edge(edge.node1.node_id, edge.node2.node_id, edge_obj=edge)
    return global_graph




def main():
    model = nn.Sequential(
        nn.Conv2d(3,4,3), # 4x3x3x3
    )
    global_graph = seq_to_net(model)
    print("Global graph nodes", global_graph.nodes)
    for node in global_graph.nodes:
        print(global_graph.nodes[node]['node_obj'])
    print("Global graph edges", len(global_graph.edges))

if __name__ == '__main__':
    main()


