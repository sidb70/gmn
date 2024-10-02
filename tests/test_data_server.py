import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import unittest
from experiments.generate_train_data import train_save_to_azure
from azure.core.exceptions import ResourceNotFoundError

# from resources import azure_files
from resources import AzureDatasetClient


class TestGenerateData(unittest.TestCase):

    @unittest.skip("skip")
    def test_upload_load_delete_tensor(self):

        client = AzureDatasetClient()

        tensors = [torch.rand(n, n, 1) for n in range(1, 100)]

        path = "test/tensor.pt"

        client._save_torch_object(tensors, path)

        loaded_tensors = client._fetch_pt_file(path)

        for t1, t2 in zip(tensors, loaded_tensors):
            self.assertTrue(torch.allclose(t1, t2))

        client._delete_file(path)

        # ensure the file is deleted
        with self.assertRaises(ResourceNotFoundError):
            client._fetch_pt_file(path)

    # @unittest.skip("skip")
    def test_train_save_to_azure(self):

        client = AzureDatasetClient("test/test-hpo")

        client.delete_dataset()

        n_architectures = 10
        train_save_to_azure(client, n_architectures=n_architectures)

        features, accuracies = client.fetch_dataset()

        self.assertEqual(len(features), len(accuracies))
        self.assertGreater(len(features), 0)


if __name__ == "__main__":
    unittest.main()
