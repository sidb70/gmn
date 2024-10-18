import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch
from train.gmn_trainer import GMNEdgePredictorTrainer
from resources import LocalFileClient, HPOExperimentClient, LocalFileClient, FixedHPExperimentClient
from train.utils import split
from models.mpnn_models import EdgePredictorMPNN
from argparse import ArgumentParser



DEVICE = torch.device("cuda")

torch.manual_seed(0)
if __name__ == "__main__":
    """
    Train gmn on random CNNs trained with random hyperparameters
    """

    args = ArgumentParser()
    args.add_argument("--results_dir", type=str, default="data/fixed_hp_result", help="Directory to save results")
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
    args.add_argument("--samples", type=int, default=-1)
    args = args.parse_args()
    
    client = FixedHPExperimentClient(LocalFileClient('/mnt/home/bhatta70/Documents/gmn/gmn/data/fixed_hp_data'))
    n_samples = args.samples
    features, labels = client.read_dataset(n_samples)
    print("Loaded ", len(features), "features and ", len(labels), "labels")

    print("labels", labels)
    # hpo_gmn = train_hpo(args, features, labels)

    results_dir = args.results_dir
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    valid_size = args.valid_size
    test_size = args.test_size
    num_epochs = args.epochs
    lr = args.lr

    os.makedirs(results_dir, exist_ok=True)

    
    model = EdgePredictorMPNN(hidden_dim).to(DEVICE)   
    

    trainer = GMNEdgePredictorTrainer(model, DEVICE)

    trainer.train(features, labels, 100, batch_size, lr, valid_size=valid_size, test_size=test_size)
