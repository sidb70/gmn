from enum import Enum
import networkx as nx
from pydantic import BaseModel
from typing import Tuple
import json
import os
# ----------------- Enums -----------------

class LayerType(Enum):
    '''
    Enum for the type of layer in a neural network. Not used for GNN, just for graph representation
    Node features encapulate the layer type, so this is not strictly necessary (likely)
    '''
    INPUT = -1
    OUTPUT = -1
    LINEAR = 1
    CONV = 2
    NORM = 3
    RELU = 4
    
class NodeType(Enum):
    '''
    Node types for param graph
    - Will be used within the node features
    - does not distinguish between weight/bias. can encode that in edge type
    '''
    INPUT = -1
    OUTPUT = -1
    NON_PARAMETRIC = -1 # relu, softmax, etc
    LINEAR = 1 # for both lin weight and lin bias (weight/bias is encoded in edge type)
    CONV = 2 # for both conv weight and conv bias
    NORM = 3

class EdgeType(Enum):
    '''
    Edge types for param graph
    Will be used within the edge features
    '''
    NONPARAMETRIC = -1 # relu, softmax, etc
    LIN_WEIGHT = 1
    LIN_BIAS = 2
    CONV_WEIGHT = 3
    CONV_BIAS = 4
    NORM_GAMMA = 5
    NORM_BETA = 6

# ----------------- Features -----------------
class NodeFeatures(BaseModel):
    layer_num: int # layer number in the network
    rel_index: int # index of the node within the layer
    node_type: NodeType 
    def serialize(self) -> dict:
        '''
        Serialize the node features into a dictionary

        Returns:
        - dict: Serialized node features
        '''
        ser = {
            'layer_num': self.layer_num,
            'rel_index': self.rel_index,
            'node_type': self.node_type.value
        }
        return ser

class EdgeFeatures(BaseModel):
    weight: float 
    layer_num: int # layer number in the network
    edge_type: EdgeType
    pos_encoding_x: int # x positional encoding of this parameter within the conv layer
    pos_encoding_y: int # y positional encoding of this parameter within the conv layer
    pos_encoding_depth: int # which layer of the conv cube this parameter is in
    def serialize(self) -> dict:
        '''
        Serialize the edge features into a dictionary

        Returns:
        - dict: Serialized edge features
        '''
        ser = {
            'weight': self.weight,
            'layer_num': self.layer_num,
            'edge_type': self.edge_type.value,
            'pos_encoding_x': self.pos_encoding_x,
            'pos_encoding_y': self.pos_encoding_y,
            'pos_encoding_depth': self.pos_encoding_depth
        }
        return ser
# ----------------- Base Classes -----------------
class ParameterGraph: pass
class Layer(ParameterGraph): pass
class Node: pass
class Edge: pass

class ParameterGraph(nx.MultiDiGraph):
    '''
    Base class for parameter graph
    - Inherits from networkx Graph
    - Will be used to represent the parameter graph
    '''
    def __init__(self) -> None:
        super().__init__()
    def serialize(self) -> dict:
        '''
        Serialize the graph into a dictionary
        '''
        nodes = [{'id': node_id, 'features': node_obj.features.serialize()} for node_id, node_obj in self.nodes(data='node_obj')]
        edges = [{'source': source, 'target': target, 'features': edge_obj.features.serialize()} for source, target, edge_obj in self.edges(data='edge_obj')]
        ser = {
            'nodes': nodes,
            'links': edges
        }
        return ser
    def to_json(self) -> str:
        '''
        Serialize the graph into a JSON string
        '''
        return json.dumps(self.serialize())
    def save(self, path: str) -> None:
        '''
        Save the graph to a JSON file

        Args:
        - path (str): Path to save the file
        '''
        if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            json.dump(self.serialize(), f)
class NetworkLayer(ParameterGraph):
    '''
    Represents a layer in the network
    - Contains nodes and edges
    - is a Graph
    '''
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
    def get_nodes(self) -> set[Node]:
        return self.nodes
    def get_edges(self) -> set[Edge]:
        return self.edges
    def get_node_ids(self) -> list[int]:
        return sorted([node.node_id for node in self.nodes])
    def __str__(self) -> str:
        return f"Layer {self.layer_num}: Type: {self.layer_type} with {len(self.nodes)} nodes and {len(self.edges)} edges"
    def __repr__(self) -> str:
        return self.__str__()
class Edge:
    connection_nodes: Tuple[Node, Node]
    features: EdgeFeatures
    def __init__(self, connection_nodes: tuple[Node], features: EdgeFeatures) -> None:
        self.node1, self.node2 = connection_nodes
        self.features = features
    def serialize(self) -> dict:
        '''
        Serialize the edge into a dictionary
        '''
        ser = {
            'connection_nodes': (self.node1.node_id, self.node2.node_id), 
            'features': self.features.serialize()
        }
        return ser
    def __str__(self) -> str:
        return f"Edge from {self.node1.node_id} to {self.node2.node_id} with features {self.features}"
    def __repr__(self) -> str:
        return self.__str__()

class Node:
    node_id: int
    features: NodeFeatures
    def __init__(self, node_id: int, features: NodeFeatures) -> None:
        '''
        Node in the network
        - Node ID is unique to the node
        - Features contain information about the node
        '''
        self.node_id = node_id
        self.features = features
    def serialize(self) -> dict:
        '''
        Serialize the node into a dictionary
        '''
        ser = {
            'node_id': self.node_id,
            'features': self.features.serialize()
        }
        return ser

    def __str__(self) -> str:
        return f"Node {self.node_id} with features {self.features}"
    def __repr__(self) -> str:
        return self.__str__()
    
if __name__=='__main__':
    testnode = Node(node_id=0, features=NodeFeatures(layer_num=0, rel_index=0, node_type=NodeType.INPUT))
    testedge = Edge(connection_nodes=(testnode, testnode), features=EdgeFeatures(weight=0.5, layer_num=0, edge_type=EdgeType.LIN_WEIGHT, pos_encoding_x=0, pos_encoding_y=0, pos_encoding_depth=0))
    # serialize

    node_json = json.dumps(testnode.serialize())
    edge_json = json.dumps(testedge.serialize())
    print("Node serialized: ",node_json)    
    print("Edge serialized: ",edge_json)