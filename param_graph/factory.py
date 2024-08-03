import torch.nn as nn
from .graph_types import (
    NetworkLayer,
    Node,
    NodeType,
    LayerType,
    NodeFeatures,
    EdgeFeatures,
    Edge,
    EdgeType,
    PARAMETRIC_LAYERS,
    BIAS_NODE_TYPES,
    get_module_type
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
        prev_layer: NetworkLayer = kwargs['prev_layer']
        layer_type = kwargs['layer_type']
        linear_layer = NetworkLayer(layer_num=layer_num, layer_type=layer_type)

        bias_node = Node(start_node_id, NodeFeatures(
            layer_num=layer_num, rel_index=-1, node_type=NodeType.LINEAR_BIAS))
        linear_layer.add_node(bias_node)
        start_node_id += 1

        for i, weights in enumerate(layer.weight.data):
            # weights corresponds to the weights for each input feature for the ith output neuron
            node_id = start_node_id + i
            node = Node(
                node_id=node_id,
                features=NodeFeatures(
                    layer_num=layer_num, rel_index=-1, node_type=NodeType.LINEAR)
            )
            linear_layer.add_node(node)

            # connect prev layer nodes to current node
            for in_node, weight in zip(prev_layer.nodes, weights):
                if in_node.features.node_type in BIAS_NODE_TYPES:
                    continue
                edge = Edge(
                    (in_node, node),
                    EdgeFeatures(weight=weight, layer_num=layer_num,
                                 edge_type=EdgeType.LIN_WEIGHT)
                )
                linear_layer.add_edge(edge)

            # bias to node edge
            edge = Edge(
                (bias_node, node),
                EdgeFeatures(
                    weight=layer.bias.data[i], layer_num=layer_num, edge_type=EdgeType.LIN_BIAS)
            )
            linear_layer.add_edge(edge)

        return linear_layer

    def create_norm_layer(self, module: nn.Module, layer_num: int, start_node_id: int, **kwargs) -> NetworkLayer:
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
        prev_layer = kwargs['prev_layer']
        layer_type = kwargs.get('layer_type', LayerType.NORM)
        norm_layer = NetworkLayer(layer_num=layer_num, layer_type=layer_type)

        gamma, beta = module.weight, module.bias
        bn_node = Node(start_node_id, NodeFeatures(layer_num=layer_num,
                                                   rel_index=-1,
                                                   node_type=NodeType.NORM,))
        norm_layer.add_node(bn_node)
        # connect to all nodes in the previous layer
        i = 0
        for node in prev_layer.nodes:
            if node.features.node_type in BIAS_NODE_TYPES:
                continue
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
            i += 1
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

        prev_layer = kwargs['prev_layer']
        layer_type = kwargs['layer_type']
        conv_layer = NetworkLayer(layer_num=layer_num, layer_type=layer_type)

        # iterate through the channels of the current layer, one new node per channel
        for out_channel in range(module.out_channels):
            node_id = start_node_id + out_channel
            out_channel_node = Node(node_id, NodeFeatures(layer_num=layer_num,
                                                          rel_index=-1,
                                                          node_type=NodeType.CONV))
            conv_layer.add_node(out_channel_node)
            # iterate through previous layer nodes:
            for idx, in_node_id in enumerate(prev_layer.nodes):
                kernel = module.weight[out_channel]
                j = 0  # index of the weight in the kernel
                # iterate through the weights of the current channel
                weights = kernel.flatten()
                for weight in weights:
                    edge_tup = (in_node_id, out_channel_node)
                    edge = Edge(edge_tup, EdgeFeatures(weight=weight,
                                                       layer_num=layer_num,
                                                       edge_type=EdgeType.CONV_WEIGHT,
                                                       pos_encoding_x=j % module.weight.shape[2],
                                                       pos_encoding_y=j//module.weight.shape[2],
                                                       pos_encoding_depth=out_channel))
                    conv_layer.add_edge(edge)
                    j += 1
                bias_edge = Edge((in_node_id, out_channel_node), EdgeFeatures(weight=module.bias[out_channel],
                                                                              layer_num=layer_num,
                                                                              edge_type=EdgeType.CONV_BIAS))
                conv_layer.add_edge(bias_edge)

        return conv_layer

    def create_non_parametric_layer(self, module: nn.Module, layer_num: int, start_node_id: int, **kwargs) -> NetworkLayer:
        '''
        Create a non-parametric layer
        - Non-parametric layers do not have weights (ReLU, Sigmoid, etc.)
        - Each node in the layer represents a neuron in the layer

        Args:
        - layer (nn.Module): PyTorch non-parametric layer
        - layer_num (int): Layer number
        - start_node_id (int): Starting node ID
        - kwargs (dict): Additional arguments

        Returns:
        - NetworkLayer: Non-parametric layer
        '''
        layer_type = kwargs['layer_type']
        non_parametric_layer = NetworkLayer(
            layer_num=layer_num, layer_type=layer_type)
        node = Node(start_node_id, NodeFeatures(layer_num=layer_num,
                    rel_index=0, node_type=NodeType.NON_PARAMETRIC))
        non_parametric_layer.add_node(node)
        prev_layer = kwargs['prev_layer']
        for prev_nodes in prev_layer.nodes:
            edge = Edge((node, prev_nodes), EdgeFeatures(
                weight=-1, layer_num=layer_num, edge_type=EdgeType.NON_PARAMETRIC))
            non_parametric_layer.add_edge(edge)
        return non_parametric_layer

    def module_to_graph(self, module: nn.Module, layer_num: int, start_node_id: int, **kwargs) -> NetworkLayer:
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
        if layer_num == 0:
            return self.create_input_layer(module, **kwargs)
        module_type = get_module_type(module)
        kwargs['layer_type'] = module_type
        if module_type == LayerType.LINEAR:
            return self.create_linear_layer(module, layer_num, start_node_id, **kwargs)
        elif module_type == LayerType.NORM:
            return self.create_norm_layer(module, layer_num, start_node_id, **kwargs)
        elif module_type == LayerType.CONV:
            return self.create_conv_layer(module, layer_num, start_node_id, **kwargs)
        elif module_type == LayerType.NON_PARAMETRIC:
            return self.create_non_parametric_layer(module, layer_num, start_node_id, **kwargs)
        else:
            raise ValueError(f"Layer type {type(module)} not yet supported")
