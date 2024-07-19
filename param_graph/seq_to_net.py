from .factory import LayerFactory
import torch.nn as nn
from .graph_types import ParameterGraph, LayerType
from .graph_types import (
    PARAMETRIC_LAYERS,
)


def seq_to_nx(seq: nn.Sequential) -> ParameterGraph:
    '''
    Convert a PyTorch sequential model to a ParameterGraph

    Args:
    - seq (nn.Sequential): PyTorch sequential model

    Returns:
    - MultiDiGraph: Global graph
    '''
    layer_factory = LayerFactory()
    global_graph = ParameterGraph()
    # create first layer
    layer_num = 0
    starting_node_id=0
    prev_layer = None
    for module in seq:
        if type(module) in [nn.Flatten, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.Identity]:
            continue
        layer = layer_factory.module_to_graph(module, layer_num=layer_num, start_node_id=starting_node_id, prev_layer=prev_layer)
        for node in layer.get_nodes():
            global_graph.add_node(node.node_id, node_obj=node)
            global_graph.nodes[node.node_id]['subset'] = layer.layer_num
        for edge in layer.get_edges():
            global_graph.add_edge(edge.node1.node_id, edge.node2.node_id, edge_obj=edge)
        if layer.layer_type in PARAMETRIC_LAYERS + [LayerType.INPUT]:
            prev_layer = layer
        layer_num += 1
        starting_node_id = max(layer.get_node_ids()) + 1
    return global_graph

