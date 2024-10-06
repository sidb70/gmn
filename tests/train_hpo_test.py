import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import unittest
from resources import HPOExperimentClient, LocalFileClient, AzureFileClient


class TrainHPOTest(unittest.TestCase):

    def test_train_hpo(self):

        data_client = HPOExperimentClient(AzureFileClient("hpo-data"))

        features, labels = data_client.read_dataset()

        print(len(features), len(labels))

        print(
            features[0].node_feats.shape,
            features[0].edge_indices.shape,
            features[0].edge_feats.shape,
        )


if __name__ == "__main__":
    unittest.main()
