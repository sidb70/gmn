import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import unittest
import torch
import json
from resources.dataset_clients import HPOExperimentClient
from resources.file_clients import LocalFileClient, AzureFileClient, FileClient


class TestTrainSaveLocally(unittest.TestCase):

    def test_save_load_tensor_local(self):

        file_clients: list[FileClient] = [
            LocalFileClient("data/test-file-client"),
            AzureFileClient("test/test-file-client"),
        ]

        for file_client in file_clients:
            tensors = [torch.rand(4, 3)]
            json_data = {"tensor": tensors}

            file_client.obj_to_pt_file(tensors, "subdir/test_tensor.pt")
            file_client.obj_to_pt_file(json_data, "subdir/test_json.pt")

            read_tensors = file_client.read_pt_file("subdir/test_tensor.pt")
            read_json = file_client.read_pt_file("subdir/test_json.pt")

            self.assertTrue(torch.equal(tensors[0], read_tensors[0]))
            self.assertTrue(torch.equal(json_data["tensor"][0], read_json["tensor"][0]))

            file_client.delete_directory("nonexistent")

            file_client.delete_directory()


if __name__ == "__main__":
    unittest.main()
