from enum import Enum
from numpy import ndarray
import networkx as nx

# ----------------- Enums -----------------
class LayerType(Enum):
    RELU = 1
    LINEAR = 2
    CONV = 3
    NORM = 4

class NodeType(Enum):
    INPUT = -1
    OUTPUT = -1
    WEIGHT = 1
    BIAS = 2
    CHANNEL = 3
    BNORM = 4

class EdgeType(Enum):
    NONPARAMETRIC = -1
    LIN_WEIGHT = 1
    LIN_BIAS = 2
    CONV_WEIGHT = 3
    CONV_BIAS = 4
    NORM_GAMMA = 5
    NORM_BETA = 6


# ----------------- Base Classes -----------------
class Graph: pass
class Layer(Graph): pass
class Node: pass
class Edge: pass

class Graph(nx.Graph):
    def __init__(self) -> None:
        super().__init__()
    def add_node(self, node: Node) -> None:
        super().add_node(node.node_id, node)
    def add_edge(self, edge: Edge) -> None:
        super().add_edge(edge.node1.node_id, edge.node2.node_id, edge)
    
class NetworkLayer(Graph):
    def __init__(self, layer_num: int, layer_type: LayerType) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.layer_type = layer_type
        self.nodes = set()
        self.edges = set()
    def add_node(self, node: Node) -> None:
        self.nodes.add(node)
    def add_edge(self, edge: Edge) -> None:
        self.edges.add(edge)

class Edge:
    def __init__(self, connection_nodes: tuple[Node], features: ndarray) -> None:
        self.node1, self.node2 = connection_nodes
        self.features = features

class Node:
    def __init__(self, node_id: int, features: ndarray) -> None:
        self.id = node_id
        self.features = features


