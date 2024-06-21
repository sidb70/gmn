import torch.nn as nn
from graph_types import NetworkLayer, Node, NodeType, LayerType, NodeFeatures


class LayerFactory:
    '''
    Factory class to create network layers from PyTorch layers
    '''
    def __init__(self) -> None:
        return
    def create_input_layer(self, layer: nn.Module, **kwargs) -> NetworkLayer:
        '''
        Create the input layer of the network
        - The input layer is the first layer of the network
        - It contains input nodes for each input feature
        
        Args:
        - layer (nn.Module): PyTorch layer
        - kwargs (dict): Additional arguments
        
        Returns:
        - NetworkLayer: Input layer
        '''
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
        '''
        Create a linear layer
        - Linear layers are fully connected layers
        - Each node in the layer represents a neuron in the layer
        
        Args:
        - layer (nn.Linear): PyTorch linear layer
        - layer_num (int): Layer number
        - start_node_id (int): Starting node ID
        - kwargs (dict): Additional arguments which must include the previous layer
        
        Returns:
        - NetworkLayer: Linear layer
        '''
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
        '''
        Create a normalization layer
        - Represented by one node in the network
        - each edge has a feature indicating the normalization parameter (e.g. gamma, beta, )

        Args:
        - layer (nn.BatchNorm1d): PyTorch normalization layer
        - layer_num (int): Layer number
        - start_node_id (int): Starting node ID
        - kwargs (dict): Additional arguments which must include the previous layer

        Returns:
        - NetworkLayer: Normalization layer
        '''
        try:
            prev_layer = kwargs['prev_layer']
        except KeyError:
            raise ValueError("Previous layer must be provided to create norm layer")
        
        raise NotImplementedError("Norm layer creation not yet implemented")
    def create_layer(self, layer: nn.Module, layer_num: int, start_node_id: int, **kwargs) -> NetworkLayer:
        '''
        Create a network layer from a PyTorch layer
        - Dispatches to the appropriate layer creation method based on the layer type
        
        Args:
        - layer (nn.Module): PyTorch layer
        - layer_num (int): Layer number
        - start_node_id (int): Starting node ID
        - kwargs (dict): Additional arguments
        
        Returns:
        - NetworkLayer: Network layer
        '''
        if layer_num==0:
            return self.create_input_layer(layer, **kwargs)
        elif isinstance(layer, nn.Linear):
            return self.create_linear_layer(layer, layer_num, start_node_id, **kwargs)
        elif isinstance(layer, nn.BatchNorm1d):
            return self.create_norm_layer(layer, layer_num, start_node_id, **kwargs)
        else:
            raise ValueError(f"Layer type {type(layer)} not yet supported")
    