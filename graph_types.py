from enum import Enum
from numpy import ndarray
import networkx as nx
from pydantic import BaseModel
from typing import Tuple

# ----------------- Enums -----------------

class LayerType(Enum):
    INPUT = -1
    OUTPUT = -1
    LINEAR = 1
    CONV = 2
    NORM = 3
    RELU = 4
    
class NodeType(Enum):
    INPUT = -1
    OUTPUT = -1
    NON_PARAMETRIC = -1
    WEIGHT = 1
    BIAS = 2
    CONV = 3
    NORM = 4

class EdgeType(Enum):
    NONPARAMETRIC = -1
    LIN_WEIGHT = 1
    LIN_BIAS = 2
    CONV_WEIGHT = 3
    CONV_BIAS = 4
    NORM_GAMMA = 5
    NORM_BETA = 6

# ----------------- Features -----------------
class NodeFeatures(BaseModel):
    layer_num: int
    rel_index: int
    node_type: NodeType

class EdgeFeatures(BaseModel):
    weight: float
    layer_num: int
    edge_type: EdgeType
    pos_encoding_x: int
    pos_encoding_y: int
    pos_encoding_depth: int

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
    layer_num: int
    layer_type: LayerType
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
    def get_node_ids(self) -> list[int]:
        return sorted([node.node_id for node in self.nodes])
    def __str__(self) -> str:
        return f"Layer {self.layer_num}: {self.layer_type} with {len(self.nodes)} nodes and {len(self.edges)} edges"
    def __repre__(self) -> str:
        return self.__str__()
class Edge:
    connection_nodes: Tuple[Node, Node]
    features: EdgeFeatures
    def __init__(self, connection_nodes: tuple[Node], features: EdgeFeatures) -> None:
        self.node1, self.node2 = connection_nodes
        self.features = features

class Node:
    node_id: int
    features: NodeFeatures
    def __init__(self, node_id: int, features: NodeFeatures) -> None:
        self.node_id = node_id
        self.features = features