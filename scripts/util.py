from resources import LocalFileClient, AzureFileClient, FileClient, HPOExperimentClient

import argparse


def parse_data_config_args(default_directory: str = "") -> HPOExperimentClient:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--filesystem",
        type=str,
        choices=["local", "azure"],
        default="local",
    )
    parser.add_argument("-o", "--results_dir", type=str, default=default_directory)

    args = parser.parse_args()

    if args.filesystem == "local":
        file_client = LocalFileClient(args.results_dir)
    elif args.filesystem == "azure":
        file_client = AzureFileClient(args.results_dir)

    return HPOExperimentClient(file_client)
