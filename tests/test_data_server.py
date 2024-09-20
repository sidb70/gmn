import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import unittest
from resources.data import upload_torch_tensor, load_pt_file, delete_file
from azure.core.exceptions import ResourceNotFoundError

class TestGenerateData(unittest.TestCase):

    def test_upload_load_delete_tensor(self):

        tensors = [torch.rand(n, n, 1) for n in range(1, 100)]

        path = 'test/t.pt'

        upload_torch_tensor(tensors, path)

        loaded_tensors = load_pt_file(path)

        for t1, t2 in zip(tensors, loaded_tensors):
            self.assertTrue(torch.allclose(t1, t2))

        delete_file(path)

        # ensure the file is deleted
        with self.assertRaises(ResourceNotFoundError):
            load_pt_file(path)



if __name__ == '__main__':
    unittest.main()

