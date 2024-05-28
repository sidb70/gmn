from typing import Mapping
import torch
import torch.nn as nn
import networkx as nx
from enum import Enum
import matplotlib.pyplot as plt


# ----------------- Enums -----------------
class LayerType(Enum):
    RELU = 1
    LINEAR = 2
    CONVOLUTION = 3
class NodeType(Enum):
    WEIGHT = 1
    BIAS = 2
class EdgeType(Enum):
    NONPARAMETRIC = 1
    PARAMETRIC = 2


# ----------------- Classes -----------------
class Graph: pass
class Layer(Graph): pass
class Node: pass
class Edge: pass

# ----------------- Graph -----------------
class Graph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.layers = list()
    def add_edge(self, edge: Edge):
        super().add_edge(edge.node1, edge.node2)
    def add_layer(self, layer: Layer):
        self.layers.append(layer)

# ----------------- Layers -----------------
class Layer:
    def __init__(self, parent_graph: Graph,index: int, in_nodes: set, input_dim: int, output_dim: int):
        self.parent_graph = parent_graph
        self.layer = torch.nn.Module()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_nodes = in_nodes
        self.index = index
        self.layer_type = None
        self.nodes = list()
        self._create_layer()
    def add_node(self, node_index: int, type: NodeType):
        node = Node(parent_layer=self, type=type, node_index=node_index)
        self.nodes.append(node)
        self.parent_graph.add_node(node)
        # set subset_key to layer index
        self.parent_graph.nodes[node]['subset_key'] = self.index
    def add_edge(self, node1: Node, node2: Node, edge_type: EdgeType, positional_encoding=None, weight=None):
        edge = Edge(node1=node1, node2=node2, edge_type=edge_type, positional_encoding=positional_encoding, weight=weight)
        self.parent_graph.add_edge(edge)
    def get_param_nodes(self):
        return {node for node in self.nodes if node.type==NodeType.WEIGHT}
    def _create_layer(self):
        pass

class LinearLayer(Layer):
    def __init__(self, parent_graph: Graph, index: int, input_dim: int, output_dim: int,in_nodes: set=set()):
        super().__init__(parent_graph=parent_graph, index=index, in_nodes=in_nodes, input_dim=input_dim, output_dim=output_dim)
        self.layer_type = LayerType.LINEAR

    def _create_layer(self):
        self.layer = nn.Linear(self.input_dim, self.output_dim)
        # init nodes
        for i in range(self.output_dim):
            self.add_node(node_index=i, type=NodeType.WEIGHT)
        self.add_node(node_index=self.output_dim, type=NodeType.BIAS)
  
        # attach to all in nodes
        for node in self.get_param_nodes():
            # attach to bias
            bias_node = self.nodes[self.output_dim]
            weight = self.layer.bias[node.index].item()
            self.add_edge(bias_node, node, edge_type=EdgeType.PARAMETRIC, weight=weight)
            for in_node in self.in_nodes:
                weight = self.layer.weight[node.index, in_node.index].item()
                
        # for in_node in self.in_nodes:
        #     for node in self.get_param_nodes():
        #         if node.type == NodeType.WEIGHT:
        #             weight = self.layer.weight[node.index, in_node.index].item()
        #         else:
        #             weight = self.layer.bias[node.index].item()
                self.add_edge(in_node, node, edge_type=EdgeType.PARAMETRIC, weight=weight)
        
class ConvolutionLayer(Layer):
    pass



# ----------------- Nodes & Edges -----------------
class Node:
    def __init__(self, parent_layer: Layer, type: NodeType, node_index: int):
        self.index = node_index
        self.type = type
        self.parent_layer = parent_layer
    def __str__(self):
        return f'{self.parent_layer.index}_{self.index}'
    def __repr__(self):
        return f'{self.parent_layer.index}_{self.index}'
class Edge:
    def __init__(self, node1: Node, node2: Node, edge_type: EdgeType, positional_encoding=None, weight=None):
        self.node1 = node1
        self.node2 = node2
        self.edge_type = edge_type
        self.positional_encoding = positional_encoding
        self.weight = weight
    

if __name__=='__main__':
    graph = Graph()
    layer1 = LinearLayer(parent_graph=graph, index=0, input_dim=2, output_dim=2)
    graph.add_layer(layer1)
    layer2 = LinearLayer(parent_graph=graph, in_nodes=layer1.get_param_nodes(), index=1, input_dim=2, output_dim=1)
    
    # num bias nodes
    num_bias_nodes = len([node for node in graph.nodes if node.type==NodeType.BIAS])
    print(f'num_bias_nodes: {num_bias_nodes}')

