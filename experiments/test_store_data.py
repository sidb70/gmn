import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from resources.data import load_pt_file, upload_torch_tensor
import torch

if __name__ == "__main__":
    t0 = torch.tensor(1)
    upload_torch_tensor(t0, 'test/test1/t.pt')
    t1 = load_pt_file('test/test1/t.pt')
    print(t1)

