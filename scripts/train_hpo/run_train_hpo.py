import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from train.gmn_trainer import GMNHPOTrainer
from resources import LocalFileClient, HPOExperimentClient, LocalFileClient
from config import local_hpo_data_dir
from train.utils import split
from models.mpnn_models import HPOMPNN
from argparse import ArgumentParser



DEVICE = torch.device("cuda")

torch.manual_seed(0)
if __name__ == "__main__":
    """
    Train gmn on random CNNs trained with random hyperparameters
    """

    args = ArgumentParser()
    args.add_argument("--results_dir", type=str, default="data/hpo_result", help="Directory to save results")
    args.add_argument("--node_feat_dim", type=int, default=3)
    args.add_argument("--edge_feat_dim", type=int, default=6)
    args.add_argument("--node_hidden_dim", type=int, default=32)
    args.add_argument("--edge_hidden_dim", type=int, default=32)
    args.add_argument("--hidden_dim", type=int, default=16)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--valid_size", type=float, default=0.1)
    args.add_argument("--test_size", type=float, default=0.1)
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--lr", type=float, default=0.001)
    args = args.parse_args()
    client = HPOExperimentClient(LocalFileClient('/mnt/home/bhatta70/Documents/gmn/gmn/data/fixed_hp_data'))
    dataset = client.read_dataset()

    features, labels = client.read_dataset()
    print("Loaded ", len(features), "features and ", len(labels), "labels")

    # hpo_gmn = train_hpo(args, features, labels)

    results_dir = args.results_dir
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    valid_size = args.valid_size
    test_size = args.test_size
    num_epochs = args.epochs
    lr = args.lr

    os.makedirs(results_dir, exist_ok=True)

    
    model = HPOMPNN(hidden_dim, hpo_dim=len(features[0][-1])).to(DEVICE)

    trainer = GMNHPOTrainer(model, DEVICE, hpo_grad_steps=0)

    trainer.train(features, labels, num_epochs, batch_size, lr, valid_size=valid_size, test_size=test_size)
