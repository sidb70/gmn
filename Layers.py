import torch
import torch.nn as nn
import networkx as nx
from enum import Enum

# Enums
class NodeType(Enum): # each node can either be a weight or a bias
    WEIGHT = 1
    BIAS = 2

 # ----------------- Graph -----------------
class Graph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.layers = []
 # ----------------- Layers -----------------
class Layer:
    def __init__(self, parent_graph: Graph):
        self.parent_graph = parent_graph
        self.nodes = []
        self.prev_layer = None
    def create_nodes(): pass

class GroupEquivariantLayer(Layer):
    '''
    Appendix A of the paper -- Equivariant Linear Layers
    Convolutions are a special case of equivariant linear layers where num_channels = num_kernels
    Linear layers are a special case of equivariant linear layers where num_channels = 1
    '''
    def __init__(self, parent_graph: Graph, input_size: int, output_size: int, num_channels: int):
        super().__init__(parent_graph)
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
# ----------------- Nodes -----------------
class Node:
    def __init__(self, parent_layer: Layer, node_type: NodeType):
        self.parent_layer = parent_layer
        self.layer_index = parent_layer.index
        self.node_index = len(parent_layer.nodes)
        self.type = node_type
    def add_edge(self, node):
        pass

class LinearNode(Node):
    def __init__(self, parent_layer: Layer):
        super().__init__(parent_layer)
        self.type = type
        self.edge = []
        
# ----------------- Edges -----------------
class Edge:
    def __init__(self, node1: Node, node2: Node):
        self.node1 = node1
        self.node2 = node2
        self.edge_weight=None
        self.positional_encoding=None


node = Node()