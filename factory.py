# Purpose: Factory class to create network layers from PyTorch layers

import torch.nn as nn
from graph_types import NetworkLayer, Node, NodeType, LayerType, NodeFeatures


class LayerFactory:
    def __init__(self) -> None:
        return
    
    def create_input_layer(self, layer: nn.Module, **kwargs) -> NetworkLayer:
        if type(layer) == nn.Linear:
            num_input_nodes = layer.in_features
        elif type(layer) == nn.Conv2d:
            num_input_nodes = layer.in_channels
        else:
            raise ValueError(f"Layer type {type(layer)} not yet supported")
        input_layer = NetworkLayer(layer_num=0, layer_type=LayerType.INPUT)
        for i in range(num_input_nodes):
            node_features = NodeFeatures(layer_num=0, 
                                        rel_index=i, 
                                        node_type=NodeType.INPUT)
            node = Node(i, node_features)
            input_layer.add_node(node)
        return input_layer
    def create_linear_layer(self, layer: nn.Linear, layer_num: int, start_node_id: int, **kwargs) -> NetworkLayer:
        try:
            prev_layer = kwargs['prev_layer']   
        except KeyError:
            raise ValueError("Previous layer must be provided to create linear layer")
        raise NotImplementedError("Linear layer creation not yet implemented")
    def create_norm_layer(self, 
                          layer: nn.BatchNorm1d, 
                          layer_num: int, 
                          start_node_id: int, 
                          **kwargs) -> NetworkLayer:
        try:
            prev_layer = kwargs['prev_layer']
        except KeyError:
            raise ValueError("Previous layer must be provided to create norm layer")
        
        raise NotImplementedError("Norm layer creation not yet implemented")
    def create_layer(self, layer: nn.Module, layer_num: int, start_node_id: int, **kwargs) -> NetworkLayer:
        if layer_num==0:
            return self.create_input_layer(layer, **kwargs)
        elif type(layer) == nn.Linear:
            return self.create_linear_layer(layer, layer_num, start_node_id, **kwargs)
        elif type(layer) == nn.BatchNorm1d:
            return self.create_norm_layer(layer, layer_num, start_node_id, **kwargs)
        else:
            raise ValueError(f"Layer type {type(layer)} not yet supported")
    