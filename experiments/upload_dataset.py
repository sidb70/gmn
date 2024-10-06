import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse

from resources import HPOExperimentClient, AzureFileClient

parser = argparse.ArgumentParser(description="Upload dataset to the server")
parser.add_argument("from_path", type=str, help="Local path to the dataset")
parser.add_argument("to_path", type=str, help="Path on Azure")
parser.add_argument(
    "-d", "--delete-existing", action="store_true", help="Overwrite existing dataset"
)

if __name__ == "__main__":
    args = parser.parse_args()

    from_path = args.from_path
    to_path = args.to_path

    dataset_client = HPOExperimentClient(AzureFileClient(to_path))

    if args.delete_existing:
        dataset_client.file_client.delete_directory()

    dataset_client.file_client.copy_dir_from_local(from_path)
