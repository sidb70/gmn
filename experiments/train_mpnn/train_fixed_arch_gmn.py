import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from scripts.util import parse_data_config_args



"""
Train a GMN that predicts validation loss from a set of hyperparams and a fixed architecture. 
Sets all parameters to 1. 
"""


if __name__ == "__main__":
  data_client = parse_data_config_args(default_directory="data/fixed-arch_gmn_hpo")





