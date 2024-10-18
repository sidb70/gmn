import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from resources import AzureFileClient, HPOExperimentClient
import argparse

parser = argparse.ArgumentParser(description="Delete dataset from azure")
parser.add_argument("dataset_path", type=str)

if __name__ == "__main__":

    args = parser.parse_args()
    dataset_path = args.dataset_path

    client = HPOExperimentClient(AzureFileClient(dataset_path))
    client.file_client.delete_directory()
