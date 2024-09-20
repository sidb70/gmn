import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import unittest
from resources.data import upload_torch_tensor, load_pt_file

class TestGenerateData(unittest.TestCase):

    def test_upload_load_tensor(self):

        tensor = torch.tensor(1)

        path = 'test/t.pt'

        upload_torch_tensor(tensor, path)

        loaded_tensor = load_pt_file(path)

        self.assertEqual(tensor, loaded_tensor)


if __name__ == '__main__':
    unittest.main()

