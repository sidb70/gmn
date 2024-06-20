from enum import Enum
from numpy import ndarray


# ----------------- Enums -----------------
class LayerType(Enum):
    RELU = 1
    LINEAR = 2
    CONV = 3
    NORM = 4

class NodeType(Enum):
    WEIGHT = 1
    BIAS = 2
    CHANNEL = 3
    BNORM = 4
    LNORM = 5
    GNORM = 6

class EdgeType(Enum):
    NONPARAMETRIC = 1
    PARAMETRIC = 2


# ----------------- Base Classes -----------------
class Graph: pass
class Layer(Graph): pass
class Node: pass
class Edge: pass

class Layer(Graph):
    def __init__(self):
        super().__init__()
        self.nodes = set()
        self.edges = set()
    def add_node(self, node: Node) -> None:
        self.nodes.add(node)
    def add_edge(self, edge: Edge) -> None:
        self.edges.add(edge)

class Edge:
    def __init__(self, connection_nodes: tuple[int], features: ndarray) -> None:
        self.node1, self.node2 = connection_nodes
        self.features = features

class Node:
    def __init__(self, index: int, features: ndarray) -> None:
        self.index = index
        self.features = features


