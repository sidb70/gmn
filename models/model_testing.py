import sys
sys.path.append('/mnt/home/bhatta70/Documents/gmn/gmn/')
from mpnn_models import EdgePredictorMPNN
import torch


mpnn = EdgePredictorMPNN(hidden_dim=6)


print(mpnn)

