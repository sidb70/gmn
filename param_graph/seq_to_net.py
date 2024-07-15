from .factory import LayerFactory
import torch.nn as nn
from .graph_types import ParameterGraph, NetworkLayer
from pprint import pprint
from .visualize import draw_graph
from .graph_types import (
    PARAMETRIC_LAYERS,
)
from typing import List

def seq_to_net(seq: nn.Sequential) -> ParameterGraph:
    '''
    Convert a PyTorch sequential model to a list of NetworkLayer objects

    Args:
    - seq (nn.Sequential): PyTorch sequential model

    Returns:
    - MultiDiGraph: Global graph
    '''
    layer_factory = LayerFactory()
    layers: List[NetworkLayer] = []

    # create input layer
    prev_layer = layer_factory.create_layer(module=seq[0], layer_num=0, start_node_id=0)
    # prev_layer is the last parametric layer
    layers.append(prev_layer)

    for layer_num, module in enumerate(seq, 1):
        if type(module) in [nn.Flatten, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.Identity]:
            continue
        node_id = max(layers[-1].get_node_ids()) + 1
        layer = layer_factory.create_layer(module, layer_num=layer_num, start_node_id=node_id, prev_layer=prev_layer)
        layers.append(layer)
        if layer.layer_type in PARAMETRIC_LAYERS:
            prev_layer = layer

    global_graph = ParameterGraph()
    for layer in layers:
        for node in layer.get_nodes():
            global_graph.add_node(node.node_id, node_obj=node)
            global_graph.nodes[node.node_id]['subset'] = layer.layer_num
        for edge in layer.get_edges():
            global_graph.add_edge(edge.node1.node_id, edge.node2.node_id, edge_obj=edge)
    return global_graph

def main():
    model = nn.Sequential(
        nn.Linear(4,10),
        nn.ReLU(),
    )
    global_graph = seq_to_net(model)
    # pprint("Nodes:")
    # pprint(sorted([node[1]['node_obj'] for node in global_graph.nodes(data=True)], key=lambda x: x.node_id))
    # print("Total edges: ", global_graph.number_of_edges())
    pprint("Edges:")
    pprint( [edge for edge in global_graph.edges(data=True) if edge[2]['edge_obj'].features.edge_type.value!=3])
    sequential_title = ',\n'.join([str(type(module)).split('.')[-1].strip(">'") for module in model])
    # draw_graph(global_graph, dim='3d',title=sequential_title, save_path='/Users/sidb/Development/gmn/graph-app/backend/static/graph2.html')

    #print(global_graph.to_json())
    #global_graph.save('/Users/sidb/Development/gmn/graph-app/public/test.json')
if __name__ == '__main__':
    main()


