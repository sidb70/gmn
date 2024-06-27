import torch.nn as nn
from graph_types import (
    NetworkLayer, 
    Node, 
    NodeType, 
    LayerType, 
    NodeFeatures, 
    EdgeFeatures, 
    Edge,
    EdgeType
)


class LayerFactory:
    '''
    Factory class to create network layers from PyTorch layers
    '''
    def __init__(self) -> None:
        return
    def create_input_layer(self, module: nn.Module, **kwargs) -> NetworkLayer:
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
        if type(module) == nn.Linear:
            num_input_nodes = module.in_features
        elif type(module) == nn.Conv2d:
            num_input_nodes = module.in_channels
        else:
            raise ValueError(f"Layer type {type(module)} not yet supported")
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
        
        linear_layer = NetworkLayer(layer_num=layer_num, layer_type=LayerType.LINEAR)
        # pseudo code:
        # iterate through previous layer weights
        # create a node for each weight
        # create an edge between the node and the previous layer node
        # set the edge weight to the weight value
        # set the edge feature to the weight index
        
        raise NotImplementedError("Linear layer creation not yet implemented")
    def create_norm_layer(self, 
                          module: nn.Module, 
                          layer_num: int, 
                          start_node_id: int, 
                          **kwargs) -> NetworkLayer:
        '''
        Create a normalization layer
        - Represented by one node in the network
        - each edge has a feature indicating the normalization parameter (e.g. gamma, beta, )

        Args:
        - layer (nn.Module): PyTorch normalization layer
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
        
        norm_layer = NetworkLayer(layer_num=layer_num, layer_type=LayerType.NORM)

        gamma, beta = module.weight, module.bias
        bn_node = Node(start_node_id, NodeFeatures(layer_num=layer_num, 
                                                   rel_index=0, 
                                                   node_type=NodeType.NORM,))
        norm_layer.add_node(bn_node)
        # connect to all nodes in the previous layer
        i=0
        for node in prev_layer.nodes:
            edge_tup = (bn_node, node)
            gamma_edge = Edge(edge_tup, EdgeFeatures(weight=gamma[i],
                                                            layer_num=layer_num,
                                                            edge_type=EdgeType.NORM_GAMMA,
                                                            pos_encoding_x=-1,
                                                            pos_encoding_y=-1,
                                                            pos_encoding_depth=-1))
            beta_edge = Edge(edge_tup, EdgeFeatures(weight=beta[i],
                                                              layer_num=layer_num,
                                                              edge_type=EdgeType.NORM_BETA,
                                                              pos_encoding_x=-1,
                                                              pos_encoding_y=-1,
                                                              pos_encoding_depth=-1))
            norm_layer.add_edge(gamma_edge)
            norm_layer.add_edge(beta_edge)
            i+=1
        return norm_layer
            
    def create_conv_layer(self, module: nn.Conv2d, layer_num: int, start_node_id: int, **kwargs) -> NetworkLayer:
        '''
        Create a convolutional layer
        - Each node in the layer represents a neuron in the layer
        - Each edge represents a connection between neurons
        
        Args:
        - layer (nn.Conv2d): PyTorch convolutional layer
        - layer_num (int): Layer number
        - start_node_id (int): Starting node ID
        - kwargs (dict): Additional arguments which must include the previous layer
        
        Returns:
        - NetworkLayer: Convolutional layer
        '''
        try:
            prev_layer = kwargs['prev_layer']
        except KeyError:
            raise ValueError("Previous layer must be provided to create conv layer")
        
        conv_layer = NetworkLayer(layer_num=layer_num, layer_type=2)

        
            #iterate through the channels of the current layer, one new node per channel
        for out_channel in range(module.out_channels):
            node_id = start_node_id + out_channel
            out_channel_node = Node(node_id, NodeFeatures(layer_num=layer_num, 
                                                                rel_index=out_channel, 
                                                                node_type=NodeType.CONV))
            conv_layer.add_node(out_channel_node)
            #iterate through previous layer nodes:
            for in_node in prev_layer.nodes:
                kernel = module.weight[out_channel]
                j=0  # index of the weight in the kernel
                #iterate through the weights of the current channel
                weights = kernel.flatten()
                for weight in weights:
                    edge_tup = (in_node,out_channel_node)
                    edge = Edge(edge_tup, EdgeFeatures(weight=weight,
                                                        layer_num=layer_num,
                                                        edge_type=EdgeType.CONV_WEIGHT,
                                                        pos_encoding_x=j%module.weight.shape[2], 
                                                        pos_encoding_y=j//module.weight.shape[2], 
                                                        pos_encoding_depth=out_channel))
                    conv_layer.add_edge(edge)
                    j+=1

            
                
            
        #TODO: add bias edges
        
        return conv_layer
                

        raise NotImplementedError("Conv layer creation not yet implemented")
    def create_layer(self, module: nn.Module, layer_num: int, start_node_id: int, **kwargs) -> NetworkLayer:
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
            return self.create_input_layer(module, **kwargs)
        elif isinstance(module, nn.Linear):
            return self.create_linear_layer(module, layer_num, start_node_id, **kwargs)
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            return self.create_norm_layer(module, layer_num, start_node_id, **kwargs)
        elif isinstance(module, nn.Conv2d):
            return self.create_conv_layer(module, layer_num, start_node_id, **kwargs)
        else:
            raise ValueError(f"Layer type {type(module)} not yet supported")
    