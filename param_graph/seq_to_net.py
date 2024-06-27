from factory import LayerFactory
import torch.nn as nn
import networkx as nx
from graph_types import ParameterGraph
import matplotlib.pyplot as plt
from pprint import pprint
import random

def seq_to_net(seq: nn.Sequential) -> ParameterGraph:
    '''
    Convert a PyTorch sequential model to a list of NetworkLayer objects

    Args:
    - seq (nn.Sequential): PyTorch sequential model

    Returns:
    - MultiDiGraph: Global graph
    '''
    layer_factory = LayerFactory()
    layers = []
    # create first layer
    first_layer = layer_factory.create_layer(module=seq[0], layer_num=0, start_node_id=0)
    layers.append(first_layer)
    # print(first_layer)
    layer_num=1
    for module in seq:
        node_id = max(layers[-1].get_node_ids()) + 1
        layer = layer_factory.create_layer(module, layer_num=layer_num, start_node_id=node_id, prev_layer=layers[-1])
        layers.append(layer)
        layer_num += 1
    global_graph = ParameterGraph()
    for layer in layers:
        for node in layer.get_nodes():
            global_graph.add_node(node.node_id, node_obj=node)
            global_graph.nodes[node.node_id]['subset'] = layer.layer_num
        for edge in layer.get_edges():
            global_graph.add_edge(edge.node1.node_id, edge.node2.node_id, edge_obj=edge)
    return global_graph


def draw_graph(graph: ParameterGraph):
    '''
    Draw the global graph

    Args:
    - graph (MultiDiGraph): Global graph
    '''
    pos = nx.multipartite_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    i = 0
    for u, v, data in graph.edges(data=True):
        rad = 0.05*i % 1.0 * random.choice([-1, 1])
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={rad}')
        i += 1
    plt.show()


def main():
    model = nn.Sequential(
        nn.Conv2d(3,4,3), # 4x3x3x3
        nn.BatchNorm2d(4),
    )
    global_graph = seq_to_net(model)
    pprint("Nodes:")
    pprint([node[1]['node_obj'] for node in global_graph.nodes(data=True)])
    print("Total edges: ", global_graph.number_of_edges())
    #draw_graph(global_graph)

    print(global_graph.to_json())
    global_graph.save('/Users/sidb/Development/gmn/graph-app/public/test.json')
if __name__ == '__main__':
    main()


