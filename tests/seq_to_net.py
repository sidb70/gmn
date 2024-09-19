import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import unittest
from pprint import pprint
import time
import torch
import torch.nn as nn
import numpy as np
from gmn_lim.model_arch_graph import seq_to_feats
from preprocessing.generate_nns import generate_random_cnn, RandCNNConfig


class TestSeqToNet(unittest.TestCase):

    def test_mlp_to_net(self):

        seq = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

        node_feats, edge_indices, edge_feats = seq_to_feats(seq)

        self.validate_net_feats(node_feats, edge_indices, edge_feats)


    def test_cnn_to_net(self):
        """
        CNN with avgpool
        """

        neq = nn.Sequential(
            nn.Conv2d(3, 4, 5),
            nn.ReLU(),
            nn.Conv2d(4, 6, 5),
            nn.BatchNorm2d(6),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        node_feats, edge_indices, edge_feats = seq_to_feats(neq)

        self.validate_net_feats(node_feats, edge_indices, edge_feats)

    # @unittest.skip("skip")
    def test_cnn_to_net_2(self):
        """
        CNN without avgpool
        """

        neq = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 10),
            nn.ReLU(),
        )

        node_feats, edge_indices, edge_feats = seq_to_feats(neq)

        self.validate_net_feats(node_feats, edge_indices, edge_feats)

    def test_time_random_cnn_to_net(self):

        start=time.time()
        model = generate_random_cnn(RandCNNConfig(
            in_dim=32,
            in_channels=3,
            n_classes=10,
            n_conv_layers_range=(10,15),
            n_fc_layers_range=(8,10),
            log_hidden_channels_range=(7,8),
            log_hidden_fc_units_range=(4,5),
        ))
        print("Time to create random cnn:", round(time.time()-start,5))
        print('Total params', sum(p.numel() for p in model.parameters()))
        start = time.time()
        feats = seq_to_feats(model)
        print([f.shape for f in feats])
        print("Time to create feats:", round(time.time()-start,5))


    def test_random_cnns_to_net(self):

        for _ in range(10):
            cnn = generate_random_cnn(
                RandCNNConfig(
                    log_hidden_channels_range=(2, 4), log_hidden_fc_units_range=(2, 4),
                    use_avg_pool_prob=1.0
                )
            )
            
            node_feats, edge_indices, edge_feats = seq_to_feats(cnn)

            self.validate_net_feats(node_feats, edge_indices, edge_feats)



    def validate_net_feats(self, node_feats, edge_indices, edge_feats):
        # 3 features per node
        self.assertEqual(node_feats.shape[1], 3) 
        # same number of edge indices and edge features
        self.assertEqual(edge_indices.shape[1], edge_feats.shape[0])



if __name__ == "__main__":
    unittest.main()
