import unittest
import os
import sys
from pprint import pprint

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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


    def test_cnn_to_net_1(self):
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


    def test_cnn_to_net_3(self):

        neq = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.BatchNorm2d(8),
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


    def test_random_cnns_to_net(self):

        torch.manual_seed(0)

        for _ in range(10):
            cnn = generate_random_cnn(
                RandCNNConfig(
                    log_hidden_channels_range=(2, 4), log_hidden_fc_units_range=(2, 4),
                    use_avg_pool_prob=torch.randint(0, 2, (1,)).item() == 1
                )
            )

            print(cnn)
            
            node_feats, edge_indices, edge_feats = seq_to_feats(cnn)

            self.validate_net_feats(node_feats, edge_indices, edge_feats)



    def validate_net_feats(self, node_feats, edge_indices, edge_feats):
        self.assertEqual(node_feats.shape[1], 3) # 3 features per node
        self.assertEqual(edge_indices.shape[1], edge_feats.shape[0])
        # same number of edge indices and edge features



if __name__ == "__main__":
    unittest.main()
