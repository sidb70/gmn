import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import unittest
from experiments.generate_train_data import train_save_to_azure
from azure.core.exceptions import ResourceNotFoundError
# from resources import azure_files
import resources

class TestGenerateData(unittest.TestCase):

    def test_upload_load_delete_tensor(self):

        tensors = [torch.rand(n, n, 1) for n in range(1, 100)]

        path = 'test/t.pt'

        resources.upload_torch_tensor(tensors, path)

        loaded_tensors = resources.load_pt_file(path)

        for t1, t2 in zip(tensors, loaded_tensors):
            self.assertTrue(torch.allclose(t1, t2))

        resources.delete_file(path)

        # ensure the file is deleted
        with self.assertRaises(ResourceNotFoundError):
            resources.load_pt_file(path)


    def test_train_save_to_azure(self):
        
        base_dir = 'test-hpo'
        feat_path = f'{base_dir}/features.pt'
        acc_path = f'{base_dir}/accuracies.pt'

        resources.delete_file(feat_path)
        resources.delete_file(acc_path)

        n_architectures = 10
        train_save_to_azure(base_dir='test-hpo', n_architectures=n_architectures)

        features = resources.load_pt_file(feat_path)
        accuracies = resources.load_pt_file(acc_path)

        self.assertEqual(len(features), len(accuracies), n_architectures)
                
        # ensure the file is deleted
        for file in [feat_path, acc_path]:
            resources.delete_file(file)
            with self.assertRaises(ResourceNotFoundError):
                resources.load_pt_file(file)


if __name__ == '__main__':
    unittest.main()

