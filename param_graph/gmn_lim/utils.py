# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import math
import torch
import torch.nn as nn

from .constants import NODE_TYPES, EDGE_TYPES

def make_node_feat(num_neurons, layer_num, node_type, is_hidden_neuron=False):
    ''' 
    makes node features for a layer
    num_neurons: number of neurons in the layer
    layer_num: the layer number
    node_type: the type of neuron
    is_hidden_neuron:  means whether it is an input or output neuron
    x has 3 dimensions:
    x[:, 0] is layer number
    x[:, 1] is neuron order (if the neuron is an input or output neuron of the whole network)
    x[:, 2] is the node type
    '''
    node_features = torch.zeros(num_neurons, 3, dtype=torch.long)
    node_features[:, 0] = layer_num
    if is_hidden_neuron:
        # if it is a hidden neuron, don't give an order
        node_features[:, 1] = -1
    else:
        node_features[:, 1] = torch.arange(num_neurons)
    node_features[:, 2] = node_type
    return node_features

def make_edge_attr(weights, layer_num, edge_type, conv_size=None, triplanar_size=None):
    '''
    weights is num_edges x 1
    triplanar size is of form (dim, N), where N is resolution [only for triplanar grid module]
    edge_attr has 6 dimensions
    edge_attr[:, 0] holds parameters
    edge_attr[:, 1] is layer number
    edge_attr[:, 2] is edge type
    edge_attr[:, (3,4,5)] are position in convolution kernel (if conv_size is not None)
    edge_attr[:, (3,4)] are position in triplanar grid (if triplanar_size is not None)
    '''
    edge_attr = torch.zeros(weights.shape[0], 6)
    edge_attr[:, 0] = weights[:, 0]
    edge_attr[:, 1] = layer_num
    edge_attr[:, 2] = edge_type
    edge_attr[:, 3:] = -1
    
    # encode position of convolution weights
    if conv_size is not None:
        positions = torch.zeros(conv_size)
        kernel_size = conv_size[2:]
        ndim = len(kernel_size)
        #kernel_pos = torch.arange(math.prod(kernel_size)).reshape(kernel_size)
        if ndim == 1:
            x = torch.arange(kernel_size[0])[None, None, :]
            positions[:] = x
            edge_attr[:, 3] = positions.flatten()
        if ndim == 2:
            x = torch.arange(kernel_size[0])[None, None, :, None]
            y = torch.arange(kernel_size[1])[None, None, None, :]
            positions[:] = x
            edge_attr[:, 3] = positions.flatten()
            positions[:] = y
            edge_attr[:, 4] = positions.flatten()
        if ndim == 3:
            x = torch.arange(kernel_size[0])[None, None, :, None, None]
            y = torch.arange(kernel_size[1])[None, None, None, :, None]
            z = torch.arange(kernel_size[2])[None, None, None, None, :]
            positions[:] = x
            edge_attr[:, 3] = positions.flatten()
            positions[:] = y
            edge_attr[:, 4] = positions.flatten()
            positions[:] = z
            edge_attr[:, 5] = positions.flatten()
    elif triplanar_size is not None:
        d, N = triplanar_size
        xyz_vals = torch.zeros(1,3*d,N,N, 3)
        # encode xy
        x = torch.linspace(-1, 1, steps=N)[None, None, :, None]
        y = torch.linspace(-1, 1, steps=N)[None, None, None, :]
        xyz_vals[:, :d, :, :, 0] = x
        xyz_vals[:, :d, :, :, 1] = y
        xyz_vals[:, :d, :, :, 2] = 0
        # encode yz
        y = torch.linspace(-1, 1, steps=N)[None, None, :, None]
        z = torch.linspace(-1, 1, steps=N)[None, None, None, :]
        xyz_vals[:, d:2*d, :, :, 0] = 0
        xyz_vals[:, d:2*d, :, :, 1] = y
        xyz_vals[:, d:2*d, :, :, 2] = z
        # encode zx
        z = torch.linspace(-1, 1, steps=N)[None, None, :, None]
        x = torch.linspace(-1, 1, steps=N)[None, None, None, :]
        xyz_vals[:, 2*d:, :, :, 0] = x
        xyz_vals[:, 2*d:, :, :, 1] = 0
        xyz_vals[:, 2*d:, :, :, 2] = z
        xvals = xyz_vals[..., 0]
        yvals = xyz_vals[..., 1]
        zvals = xyz_vals[..., 2]
        edge_attr[:, 3] = xvals.flatten()
        edge_attr[:, 4] = yvals.flatten()
        edge_attr[:, 5] = zvals.flatten()
        
    return edge_attr

def make_residual_feat(num_neurons, layer_num):
    edge_attr = torch.zeros(num_neurons, 6)
    edge_attr[:, 0] = 1 # all weights set to 1
    edge_attr[:, 1] = layer_num
    edge_attr[:, 2] = EDGE_TYPES['residual']
    edge_attr[:, 3:] = -1
    return edge_attr

def conv_to_graph(weight, bias, layer_num, in_neuron_idx, is_output=False, curr_idx=0, self_loops=True):
    '''
    converts a convolution layer to a parameter graph
    
    Args:
        weight (torch.Tensor): the weight tensor of the convolution layer
        bias (torch.Tensor): the bias tensor of the convolution layer
        layer_num (int): the layer number
        in_neuron_idx (torch.Tensor): the indices of the input neurons
        is_output (bool): whether the output neurons are the last layer
        curr_idx (int): the current index of the neurons
        self_loops (bool): whether to add self loops to the graph
        
    Returns:
        dict: a dictionary containing the graph parameters
            - 'input_neurons': the number of input neurons
            - 'output_neurons': the number of output neurons
            - 'in_neuron_idx': the indices of the input neurons
            - 'out_neuron_idx': the indices of the output neurons
            - 'node_feats': the node features
            - 'edge_index': the edge indices
            - 'edge_attr': the edge attributes
    '''
    # should work for Conv1d, Conv2d, Conv3d
    edge_attr = []
    edge_index = []
    
    input_neurons = weight.shape[1]
    assert input_neurons == in_neuron_idx.shape[0]
    
    feat = make_node_feat(input_neurons, layer_num, NODE_TYPES['channel'], is_hidden_neuron=(layer_num!=0))
    input_x = feat
    
    output_neurons = weight.shape[0]
    #out_neuron_idx = torch.arange(output_neurons) + feat.shape[0]
    out_neuron_idx = torch.arange(output_neurons) + in_neuron_idx.max() + 1
    if out_neuron_idx.min() < curr_idx:
        out_neuron_idx = out_neuron_idx + curr_idx - out_neuron_idx.min()
    feat = make_node_feat(output_neurons, layer_num+1, NODE_TYPES['channel'], is_output)
    other_nodes_feats = feat
    

    filter_size = math.prod(weight.shape[2:])
    edge_attr.append(make_edge_attr(
                    weight.reshape(-1, 1), layer_num, EDGE_TYPES['conv_weight'],
                    conv_size=weight.shape))
    #weight_edges = torch.cartesian_prod(in_neuron_idx, out_neuron_idx).T.repeat_interleave(filter_size, dim=1)
    # TODO: double check this
    weight_edges = torch.cartesian_prod(out_neuron_idx, in_neuron_idx).T
    temp = torch.zeros_like(weight_edges)
    temp[1], temp[0] = weight_edges[0], weight_edges[1]
    weight_edges = temp
    weight_edges = weight_edges.repeat_interleave(filter_size, dim=1)

    edge_index.append(weight_edges)

    if bias is not None: # checks if layer has bias
        if self_loops:
            edge_attr.append(make_edge_attr(
                        bias.reshape(-1, 1), layer_num, EDGE_TYPES['conv_bias']))
            weight_edges = torch.cat((out_neuron_idx[None, :],
                                      out_neuron_idx[None, :]), dim=0) # self loops
            edge_index.append(weight_edges)
        else:
            edge_attr.append(make_edge_attr(
                        bias.reshape(-1, 1), layer_num, EDGE_TYPES['conv_bias']))
            bias_node = make_node_feat(1, layer_num+1, NODE_TYPES['channel_bias'], False)
            bias_num = out_neuron_idx.max() + 1
            weight_edges = torch.cat([
                            torch.tensor([bias_num]).repeat(output_neurons)[None, :],
                            out_neuron_idx[None, :]
                            ], 0)
            other_nodes_feats = torch.cat([other_nodes_feats, bias_node], 0)
            edge_index.append(weight_edges)
        
    # does not work for when residual layer is first layer
    all_node_feats = other_nodes_feats if layer_num > 0 else torch.cat([input_x, other_nodes_feats], 0)
    
    edge_attr = torch.cat(edge_attr, dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    assert edge_attr.shape[0] == edge_index.shape[1]
    
    ret = { 'input_neurons': input_neurons,
            'output_neurons': output_neurons,
            'in_neuron_idx': in_neuron_idx,
            'out_neuron_idx': out_neuron_idx,
            'node_feats': all_node_feats,
            'edge_index': edge_index,
            'edge_attr': edge_attr}
    return ret

def linear_to_graph(weight, bias, layer_num, in_neuron_idx, is_output=False, curr_idx=0, self_loops=True, out_neuron_idx=None, label=''):
    ''' if out_neuron_idx is not None, then do not make new out neurons

    Converts a linear layer to its parameter graph representation

    Args:
        weight (torch.Tensor): the weight tensor of the linear layer
        bias (torch.Tensor): the bias tensor of the linear layer
        layer_num (int): the layer number
        in_neuron_idx (torch.Tensor): the indices of the input neurons
        is_out_neuron (bool): whether the output neurons are the last layer
        curr_idx (int): the current index of the neurons

    Returns:
        dict: a dictionary containing the graph parameters
            - 'input_neurons': the number of input neurons
            - 'output_neurons': the number of output neurons
            - 'in_neuron_idx': the indices of the input neurons
            - 'out_neuron_idx': the indices of the output neurons
            - 'node_feats': the node features with shape (num_neurons, 3)
            - 'edge_index': the edge indices with shape (2, num_edges)
            - 'edge_attr': the edge attributes with shape (num_edges, 6)

    '''
    edge_attr = []
    edge_index = []
    
    input_neurons = weight.shape[1]
    
    input_nodes_feats = make_node_feat(input_neurons, layer_num, NODE_TYPES[label + 'neuron'], is_hidden_neuron=(layer_num!=0))
    other_nodes_feats = None # for output and/or bias neurons
    if layer_num == 0:
        curr_idx += input_nodes_feats.shape[0]
    
    output_neurons = weight.shape[0]
    if out_neuron_idx is None:
        # need to add out neurons
        # out_neuron_idx = torch.arange(output_neurons) + in_neuron_idx.max() + 1
        # if out_neuron_idx.min() < curr_idx:
        #     # if adding out_neurons, need to make sure they start at curr_idx
        #     out_neuron_idx = out_neuron_idx - out_neuron_idx.min() + curr_idx
        out_neuron_idx = torch.arange(output_neurons) + curr_idx
        other_nodes_feats = make_node_feat(output_neurons, layer_num+1, NODE_TYPES[label + 'neuron'], is_hidden_neuron= (not is_output))
        curr_idx += other_nodes_feats.shape[0]
    else:
        # do not add new neurons
        pass
    
    edge_attr.append(make_edge_attr(
                    weight.reshape(-1, 1), layer_num, EDGE_TYPES['lin_weight']))
    
    # print(edge_attr[0].shape)

    weight_edges = torch.cartesian_prod(out_neuron_idx, in_neuron_idx).T
    temp = torch.zeros_like(weight_edges)
    temp[1], temp[0] = weight_edges[0], weight_edges[1]
    weight_edges = temp # 2 x num_edges. each col represents an edge, with row1 =in_neuron, row2=out_neuron

    # print(weight_edges.shape, edge_attr[0].shape, len(edge_attr))

    assert weight_edges.shape[1] == edge_attr[0].shape[0]
    edge_index.append(weight_edges)

    if bias is not None: # checks if layer has bias
        if self_loops:
            edge_attr.append(make_edge_attr(
                        bias.reshape(-1, 1), layer_num, EDGE_TYPES['lin_bias']))
            weight_edges = torch.cat((out_neuron_idx[None, :],
                                      out_neuron_idx[None, :]), dim=0) # self loops
            edge_index.append(weight_edges)
        else:
            edge_attr.append(make_edge_attr(
                        bias.reshape(-1, 1), layer_num, EDGE_TYPES['lin_bias']))
            
            bias_node = make_node_feat(num_neurons=1, layer_num=layer_num, node_type=NODE_TYPES['bias'], is_hidden_neuron=False)
            bias_node_idx = curr_idx
            curr_idx +=1
            weight_edges = torch.cat([
                            torch.tensor([bias_node_idx]).repeat(output_neurons)[None, :],
                            out_neuron_idx[None, :]
                            ], 0)
            
            edge_index.append(weight_edges)
            if other_nodes_feats is not None:
                other_nodes_feats = torch.cat([other_nodes_feats, bias_node], 0)
            else:
                other_nodes_feats = bias_node
            
    if layer_num > 0:
        all_node_feats = other_nodes_feats
    else:
        if other_nodes_feats is not None:
            all_node_feats = torch.cat([input_nodes_feats, other_nodes_feats], 0)
        else:
            all_node_feats = input_nodes_feats
        
    edge_attr = torch.cat(edge_attr, dim=0) # convert list of edge attributes to single tensor
    edge_index = torch.cat(edge_index, dim=1) # convert list of edge indices to single tensor
    
    ret = { 'input_neurons': input_neurons,
            'output_neurons': output_neurons,
            'in_neuron_idx': in_neuron_idx,
            'out_neuron_idx': out_neuron_idx,
            'node_feats': all_node_feats,
            'edge_index': edge_index,
            'edge_attr': edge_attr}
    return ret

def norm_to_graph(gamma, beta, layer_num, in_neuron_idx, is_output=False, curr_idx=0, self_loops=True, norm_type='bn'):
    # gamma, beta are both length d vectors
    edge_attr = []
    edge_index = []
    input_neurons = gamma.shape[0]
    assert input_neurons == in_neuron_idx.shape[0]
    
    feat = make_node_feat(input_neurons, layer_num, NODE_TYPES['neuron'], is_hidden_neuron=(layer_num!=0))
    input_x = feat
    
    output_neurons = gamma.shape[0]
    out_neuron_idx = in_neuron_idx.clone()
    feat = make_node_feat(input_neurons, layer_num, NODE_TYPES['neuron'], is_output)
    other_nodes_feats = feat
    
    if self_loops:
        added_neurons = 0
        all_node_feats = None
        weight_edges = torch.cat((out_neuron_idx[None, :],
                                  out_neuron_idx[None, :]), dim=0) # self loops
        edge_index.append(weight_edges)
        edge_index.append(weight_edges.clone())
    else:
        gamma_neuron = make_node_feat(1, layer_num, NODE_TYPES[f'{norm_type}_gamma'], is_output)
        beta_neuron = make_node_feat(1, layer_num, NODE_TYPES[f'{norm_type}_beta'], is_output)
        all_node_feats = torch.cat([gamma_neuron, beta_neuron], 0)
        
        gamma_num = curr_idx
        beta_num = gamma_num + 1
        weight_edges = torch.cat([
            torch.tensor([gamma_num]).repeat(output_neurons)[None, :],
            in_neuron_idx[None, :] ], 0)
        edge_index.append(weight_edges)
        
        weight_edges = torch.cat([
            torch.tensor([beta_num]).repeat(output_neurons)[None, :],
            in_neuron_idx[None, :] ], 0)
        edge_index.append(weight_edges)
        
    edge_attr.append(make_edge_attr(
                    gamma.reshape(-1, 1), layer_num, EDGE_TYPES[f'{norm_type}_gamma']))
    edge_attr.append(make_edge_attr(
                    beta.reshape(-1, 1), layer_num, EDGE_TYPES[f'{norm_type}_beta']))
    
    edge_attr = torch.cat(edge_attr, dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    
    ret = { 'input_neurons': input_neurons,
            'output_neurons': output_neurons,
            'in_neuron_idx': in_neuron_idx,
            'out_neuron_idx': out_neuron_idx,
            'node_feats': all_node_feats,
            'edge_index': edge_index,
            'edge_attr': edge_attr}
    return ret

def ffn_to_graph(weights1, biases1, weights2, biases2, layer_num, in_neuron_idx, is_output=False, curr_idx=0, self_loops=True):
    # as in PositionwiseFeedForward from Transformer
    # 2-layer MLP with residual connection
    ret1 = linear_to_graph(weights1, biases1, layer_num, in_neuron_idx, is_output=False, curr_idx=curr_idx, self_loops=self_loops)
    curr_idx += ret1['node_feats'].shape[0]
    ret2 = linear_to_graph(weights2, biases2, layer_num+1, ret1['out_neuron_idx'], is_output=is_output, curr_idx=curr_idx, self_loops=self_loops)
    
    all_node_feats = torch.cat([ret1['node_feats'], ret2['node_feats']], 0)
    residuals = torch.cat([in_neuron_idx.unsqueeze(0), 
                           ret2['out_neuron_idx'].unsqueeze(0)], 0)
    residuals_feat = make_residual_feat(in_neuron_idx.shape[0], layer_num)
    edge_index = torch.cat([ret1['edge_index'], ret2['edge_index'], residuals], 1)
    
    
    edge_attr = torch.cat([ret1['edge_attr'], ret2['edge_attr'], residuals_feat], 0)
    ret = {'input_neurons': ret1['input_neurons'],
           'output_neurons': ret2['output_neurons'],
           'in_neuron_idx': in_neuron_idx,
           'out_neuron_idx': ret2['out_neuron_idx'],
           'node_feats': all_node_feats,
           'edge_index': edge_index,
           'edge_attr': edge_attr}
    return ret

def basic_block_to_graph(params, layer_num, in_neuron_idx, is_output=False, curr_idx=0, self_loops=True):
    # TODO
    # no biases in the convolutions
    ret1 = conv_to_graph(params[0], None, layer_num, in_neuron_idx, is_output=False, curr_idx=curr_idx, self_loops=self_loops)
    if ret1['node_feats'] is not None:
        curr_idx += ret1['node_feats'].shape[0]
    middle_neuron_idx = ret1['out_neuron_idx']
    
    ret2 = norm_to_graph(params[1], params[2], layer_num, middle_neuron_idx, is_output=False, curr_idx=curr_idx, self_loops=self_loops, norm_type='bn')
    if ret2['node_feats'] is not None:
        curr_idx += ret2['node_feats'].shape[0]
    
    ret3 = conv_to_graph(params[3], None, layer_num+1, middle_neuron_idx, is_output=is_output, curr_idx=curr_idx, self_loops=self_loops)
    if ret3['node_feats'] is not None:
        curr_idx += ret3['node_feats'].shape[0]
    out_neuron_idx = ret3['out_neuron_idx']
    
    ret4 = norm_to_graph(params[4], params[5], layer_num+1, out_neuron_idx, is_output=False, curr_idx=curr_idx, self_loops=self_loops)
    if ret4['node_feats'] is not None:
        curr_idx += ret4['node_feats'].shape[0]
    
    # TODO: handle when all_node_feats is None
    all_node_feats = torch.cat([ret1['node_feats'], ret2['node_feats'],
                         ret3['node_feats'], ret4['node_feats']], 0)
    edge_index = torch.cat([ret1['edge_index'], ret2['edge_index'],
                            ret3['edge_index'], ret4['edge_index']], 1)
    edge_attr = torch.cat([ret1['edge_attr'], ret2['edge_attr'],
                           ret3['edge_attr'], ret4['edge_attr']], 0)
    # residual
    if len(params) == 9:
        # put through 1x1 conv and bn first
        ret5 = conv_to_graph(params[6], None, layer_num+1, in_neuron_idx, is_output=False, curr_idx=curr_idx, self_loops=self_loops)
        if ret5['node_feats'] is not None:
            curr_idx += ret5['node_feats'].shape[0]
        residual_neuron_idx = ret5['out_neuron_idx']
        
        ret6 = norm_to_graph(params[7], params[8], layer_num+1, residual_neuron_idx, is_output=False, curr_idx=curr_idx, self_loops=self_loops, norm_type='bn')
        if ret6['node_feats'] is not None:
            curr_idx += ret6['node_feats'].shape[0]
        residual_edge_index = torch.cat([residual_neuron_idx.unsqueeze(0),
                                         out_neuron_idx.unsqueeze(0)], 0)
        residual_edge_attr = make_residual_feat(out_neuron_idx.shape[0], layer_num)
        # TODO: handle when all_node_feats is None
        all_node_feats = torch.cat([all_node_feats, ret5['node_feats'], ret6['node_feats']], 0)
        edge_index = torch.cat([edge_index, ret5['edge_index'],
                                ret6['edge_index'], residual_edge_index], 1)
        edge_attr = torch.cat([edge_attr, ret5['edge_attr'],
                               ret6['edge_attr'], residual_edge_attr], 0)
    else:
        # correct shape, directly add
        assert len(params) == 6
        residual_edge_index = torch.cat([in_neuron_idx.unsqueeze(0),
                                         out_neuron_idx.unsqueeze(0)], 0)
        residual_edge_attr = make_residual_feat(out_neuron_idx.shape[0], layer_num)
        edge_index = torch.cat([edge_index, residual_edge_index], 1)
        edge_attr = torch.cat([edge_attr, residual_edge_attr], 0)
    
    ret = {'input_neurons': ret1['input_neurons'],
           'output_neurons': out_neuron_idx.shape[0],
           'in_neuron_idx': in_neuron_idx,
           'out_neuron_idx': out_neuron_idx,
           'node_feats': all_node_feats,
           'edge_index': edge_index,
           'edge_attr': edge_attr}
    return ret

def self_attention_to_graph(in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, layer_num, in_neuron_idx, is_output=False, curr_idx=0, self_loops=False):
    # abc
    wq, wk, wv = in_proj_weight.chunk(3)
    bq, bk, bv = in_proj_bias.chunk(3)
    
    # TODO: label neurons differently
    ret1 = linear_to_graph(wq, bq, layer_num, in_neuron_idx, is_output=False, curr_idx=curr_idx, self_loops=self_loops, label='attention_')
    middle_neuron_idx = ret1['out_neuron_idx']
    curr_idx += ret1['node_feats'].shape[0]
    ret2 = linear_to_graph(wk, bk, layer_num, in_neuron_idx, out_neuron_idx=middle_neuron_idx, is_output=False, curr_idx=curr_idx, self_loops=self_loops, label='attention_')
    curr_idx += ret2['node_feats'].shape[0]
    ret3 = linear_to_graph(wv, bv, layer_num, in_neuron_idx, out_neuron_idx=middle_neuron_idx, is_output=False, curr_idx=curr_idx, self_loops=self_loops, label='attention_')
    curr_idx += ret3['node_feats'].shape[0]
    
    ret4 = linear_to_graph(out_proj_weight, out_proj_bias, layer_num+1, middle_neuron_idx, is_output=is_output, curr_idx=curr_idx, self_loops=self_loops, label='attention_')
    curr_idx += ret4['node_feats'].shape[0]
    out_neuron_idx = ret4['out_neuron_idx']
    
    # TODO: number of heads not encoded in any way
    
    # TODO: look into residual connection, may need to add to SelfAttention
    residual_edge_index = torch.cat([in_neuron_idx.unsqueeze(0),
                                     out_neuron_idx.unsqueeze(0)], 0)
    residual_edge_attr = make_residual_feat(out_neuron_idx.shape[0], layer_num)
    
    
    
    all_node_feats = torch.cat([ret1['node_feats'], ret2['node_feats'], ret3['node_feats'], ret4['node_feats']], 0)
    edge_index = torch.cat([ret1['edge_index'], ret2['edge_index'],
                            ret3['edge_index'], ret4['edge_index'], residual_edge_index], 1)
    edge_attr = torch.cat([ret1['edge_attr'], ret2['edge_attr'],
                           ret3['edge_attr'], ret4['edge_attr'], residual_edge_attr], 0)
    
    ret = {'input_neurons': ret1['input_neurons'],
           'output_neurons': out_neuron_idx.shape[0],
           'in_neuron_idx': in_neuron_idx,
           'out_neuron_idx': out_neuron_idx,
           'node_feats': all_node_feats,
           'edge_index': edge_index,
           'edge_attr': edge_attr}
    return ret


def equiv_set_linear_to_graph(weight1, bias1, weight2, layer_num, in_neuron_idx, is_output=False, curr_idx=0, self_loops=False):
    ret1 = linear_to_graph(weight1, bias1, layer_num, in_neuron_idx, is_output=is_output, curr_idx=curr_idx, self_loops=self_loops, label='deepsets_')
    curr_idx += ret1['node_feats'].shape[0]
    out_neuron_idx = ret1['out_neuron_idx']
    ret2 = linear_to_graph(weight2, None, layer_num, in_neuron_idx, out_neuron_idx=out_neuron_idx, is_output=is_output, curr_idx=curr_idx, self_loops=self_loops, label='deepsets_')
    
    edge_index = torch.cat([ret1['edge_index'], ret2['edge_index']], 1)
    edge_attr = torch.cat([ret1['edge_attr'], ret2['edge_attr']], 0)
    
    ret = {'input_neurons': ret1['input_neurons'],
           'output_neurons': out_neuron_idx.shape[0],
           'in_neuron_idx': in_neuron_idx,
           'out_neuron_idx': out_neuron_idx,
           'node_feats': ret1['node_feats'],
           'edge_index': edge_index,
           'edge_attr': edge_attr}
    return ret


def triplanar_to_graph(tgrid, layer_num, is_output=False, curr_idx=0):
    ''' assumes xyz is concatenated to the triplanar features'''
    assert layer_num == 0, 'triplanar layer must be first layer'
    _, dimx3, N, N = tgrid.shape
    dim = dimx3 // 3
    
    xyz_idx = torch.arange(3)
    xyz_x = make_node_feat(3, layer_num, NODE_TYPES['neuron'], is_hidden_neuron=False)
    
    # make 3 * N * N nodes for input neurons (one for each spatial position)
    spatial_neuron_idx = torch.arange(3*N*N) + 3
    feat_neuron_idx = torch.arange(dim) + 3*N*N + 3
    
    edge_index = torch.cat([spatial_neuron_idx.repeat_interleave(dim).unsqueeze(0), feat_neuron_idx.repeat(3*N*N).unsqueeze(0)], 0)
    
    spatial_x = make_node_feat(3*N*N, layer_num, NODE_TYPES['triplanar'], is_hidden_neuron=False)
    neuron_x = make_node_feat(dim, layer_num, NODE_TYPES['triplanar'], is_hidden_neuron=True)
    all_node_feats = torch.cat([xyz_x, spatial_x, neuron_x], 0)
    weights = tgrid.flatten()[:, None]
    edge_attr = make_edge_attr(weights, layer_num, EDGE_TYPES['triplanar'], triplanar_size=(dim, N))
    
    in_neuron_idx = torch.cat([xyz_idx, spatial_neuron_idx], 0)
    out_neuron_idx = torch.cat([xyz_idx, feat_neuron_idx], 0)
    
    # make 3 * N * N * d edges for features
    ret = {'input_neurons': in_neuron_idx.shape[0],
           'output_neurons': out_neuron_idx.shape[0],
           'in_neuron_idx': in_neuron_idx,
           'out_neuron_idx': out_neuron_idx,
           'node_feats': all_node_feats,
           'edge_index': edge_index,
           'edge_attr': edge_attr}
    return ret
