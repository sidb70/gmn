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
from gmn_lim.encoders import NodeEdgeFeatEncoder


class EdgeModel(nn.Module):
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
        # print("called edge model")
        # print("edge forward src: ", src.shape, "dst: ", dst.shape, "edge_attr: ", edge_attr.shape)
        data = torch.cat([src, dst, edge_attr], 1)
        return self.phi_e(data)


class NodeModel(nn.Module):
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
            nn.Linear(in_dim, out_dim),
            nn.ReLU() if use_activation else nn.Identity()
        )

    def forward(self, x, edge_index, edge_attr, u, batch) -> torch.Tensor:
        '''
        Forward pass for node update (phi^v)
        Args:
        - x: [N, F_x], where N is the number of nodes.
        - edge_index: [2, E] with max entry N - 1.
        - edge_attr: [E, F_e]
        - u: [B, F_u]
        - batch: [N] with max entry B - 1.
        '''
        row, col = edge_index
        data = torch.cat([x[row], edge_attr], dim=1)
        data = self.node_mlp_1(data)
        data = scatter(data, col, dim=0, dim_size=x.size(0),
                       reduce='mean')
        data = torch.cat([x, data], dim=1)
        # print("called node model, data shape: ", data.shape)
        return self.node_mlp_2(data)


# global model not currently used by mpnn gmn
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

class BaseMPNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = NodeEdgeFeatEncoder(hidden_dim=hidden_dim)
        edge_in_dim = 3*hidden_dim
        node_in_dim = 2*hidden_dim
        self.meta_layers = nn.ModuleList(
            [
                MetaLayer(EdgeModel(edge_in_dim, hidden_dim),
                          NodeModel(node_in_dim, hidden_dim))
                for _ in range(3)
            ]
        )
        self.node_norm = nn.BatchNorm1d(hidden_dim)
        self.edge_norm = nn.BatchNorm1d(hidden_dim)
        self.regression = nn.Linear(2*hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # print("Base MPNN forward")
        x, edge_attr = self.encoder(x, edge_attr)
        for i, layer in enumerate(self.meta_layers):
            x, edge_attr, u = layer.forward(x, edge_index, edge_attr, u, batch)
            if i < len(self.meta_layers) - 1:
                # print('calling norm layers')
                x = self.node_norm(x)
                edge_attr = self.edge_norm(edge_attr)
        node_attr_readout = torch.mean(x, dim=0).unsqueeze(0)
        edge_attr_readout = torch.mean(edge_attr, dim=0).unsqueeze(0)
        # print("Node attr readout shape", node_attr_readout.shape, "edge attr readout shape", edge_attr_readout.shape)
        meta_out = torch.cat([node_attr_readout, edge_attr_readout], dim=1)
        return self.regression(meta_out)

class HPOMPNN(BaseMPNN):
    def __init__(self, hidden_dim, hpo_dim):
        super().__init__(hidden_dim)
        self.hpo_encoder = nn.Sequential(
            nn.Linear(hpo_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, edge_index, edge_attr, hpo, u=None, batch=None):
        x, edge_attr = self.encoder(x, edge_attr)
        hpo = self.hpo_encoder(hpo)
        for i, layer in enumerate(self.meta_layers):
            x, edge_attr, u = layer.forward(x, edge_index, edge_attr, u, batch)
            if i < len(self.meta_layers) - 1:
                x = self.node_norm(x)
                edge_attr = self.edge_norm(edge_attr)
        node_attr_readout = torch.mean(x, dim=0).unsqueeze(0)
        edge_attr_readout = torch.mean(edge_attr, dim=0).unsqueeze(0)
        meta_out = torch.cat([node_attr_readout, edge_attr_readout, hpo], dim=1)
        return self.mlp(meta_out)