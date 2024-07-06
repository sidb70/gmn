'''
Based on:

Relational inductive biases, deep learning, and graph networks; Battaglia et al.
https://arxiv.org/abs/1806.01261v3

Graph Metanetworks for Processing Diverse Neural Architectures; Lim et al.
https://arxiv.org/abs/2312.04501

'''
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer

class EdgeModel:
    '''
    from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MetaLayer.html
    '''
    def __init__(self, in_dim, out_dim, use_activation=True):
        super().__init__()
        self.phi_e = nn.Sequential(nn.Linear(in_dim, out_dim),
                                      nn.ReLU() if use_activation else nn.Identity())
                                   
    def forward(self, src, dst, edge_attr, u, batch):
        ''''
        Forward pass for edge update (phi^e)
        Args:
        - src, dst: [E, F_x], where E is the number of edges.
        - edge_attr: [E, F_e]
        - u: [B, F_u], where B is the number of graphs.
        - batch: [E] with max entry B - 1.
        '''
        print("inputs: ", 'src: ', src.shape, 'dst: ', dst.shape, 'edge_attr: ', edge_attr.shape)
        data = torch.cat([src, dst, edge_attr], 1)
        print("called edge model, data shape: ", data.shape)
        return self.phi_e(data)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
class NodeModel:
    '''
    from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MetaLayer.html
    '''
    def __init__(self, in_dim, out_dim, use_activation=True):
        super().__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU() if use_activation else nn.Identity()
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(in_dim , out_dim),
            nn.ReLU() if use_activation else nn.Identity()
        )
                                   
    def forward(self, node_feats, edge_indices, edge_feats, u, batch)-> torch.Tensor:
        '''
        Forward pass for node update (phi^v)
        Args:
        - x: [N, F_x], where N is the number of nodes.
        - edge_index: [2, E] with max entry N - 1.
        - edge_attr: [E, F_e]
        - u: [B, F_u]
        - batch: [N] with max entry B - 1.
        '''
        row, col = edge_indices
        data = torch.cat([node_feats[row], edge_feats], dim=1)
        data = self.node_mlp_1(data)
        data = scatter(data, col, dim=0, dim_size=node_feats.size(0),
                      reduce='mean')
        data = torch.cat([node_feats, data], dim=1)
        print("called node model, data shape: ", data.shape)
        return self.node_mlp_2(data)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

#### global model not currently used by mpnn gmn
# class GlobalModel:
#     def __init__(self, in_dim, out_dim):
#         self.global_mlp = nn.Sequential(
#             nn.Linear(in_dim, out_dim*2),
#             nn.ReLU(),
#             nn.Linear(out_dim*2, out_dim)
#         )
#     def forward(self, x, edge_index, edge_attr, u, batch):
#         '''
#         Forward pass for global update (phi^u)
        
#         # x: [N, F_x], where N is the number of nodes.
#         # edge_index: [2, E] with max entry N - 1.
#         # edge_attr: [E, F_e]
#         # u: [B, F_u]
#         # batch: [N] with max entry B - 1.
#         '''
#         out = torch.cat([
#             u,
#             scatter(x, batch, dim=0, reduce='mean'),
#         ], dim=1)
#         return self.global_mlp(out)
#     def __call__(self, *args, **kwargs):    
#         return self.forward(*args, **kwargs)

class BaseMPNN:
    def __init__(self, node_feat_dim, edge_feat_dim, node_hidden_dim, edge_hidden_dim, node_out_dim, edge_out_dim):
        super().__init__()
        edge_in_dim = 2*node_feat_dim + edge_feat_dim
        node_in_dim = node_feat_dim + edge_hidden_dim
        self.meta_layers = nn.ModuleList(
            [
                MetaLayer(EdgeModel(edge_in_dim, edge_hidden_dim), NodeModel(node_in_dim, node_hidden_dim)),
                MetaLayer(EdgeModel(2 * node_hidden_dim + edge_hidden_dim, edge_hidden_dim), NodeModel(node_hidden_dim + edge_hidden_dim, node_hidden_dim)),
                MetaLayer(EdgeModel(2 * node_hidden_dim + edge_hidden_dim, edge_out_dim), NodeModel(node_hidden_dim + edge_out_dim, node_out_dim)),
            ])
        self.node_norm = nn.BatchNorm1d(node_hidden_dim)
        self.edge_norm = nn.BatchNorm1d(edge_hidden_dim)
        

    def forward(self, x, edge_index, edge_attr, u, batch):
        for i,layer in enumerate(self.meta_layers):
            print("calling layer: ", i)
            x, edge_attr, u = layer(x, edge_index, edge_attr,  batch)
            if i < len(self.meta_layers) - 1:
                print('calling norm layers')
                x = self.node_norm(x)
                edge_attr = self.edge_norm(edge_attr)
        return x, edge_attr
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
if __name__=='__main__':
    # edge test
   
    edge_feats = torch.randn(5, 3)
    src_feats = torch.randn(5, 2)
    dest_feats = torch.randn(5, 2)
    edge_model = EdgeModel(7 + 4, 16)
    e_hat = edge_model(src_feats, dest_feats, edge_feats, torch.randn(3, 4), torch.tensor([0, 0, 1, 1, 2]))
    assert e_hat.shape == (5, 16)