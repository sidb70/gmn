import os
import io
from dotenv import load_dotenv

import torch
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.fileshare import ShareServiceClient, ShareDirectoryClient
from preprocessing.preprocessing_types import HPOFeatures
from typing import Tuple, List

from preprocessing.preprocessing_types import HPODataset

from abc import ABC, abstractmethod


AZURE_FILESHARE_NAME = "data"


class AzureDatasetClient:
    """
    Utility class for interacting with an HPO dataset stored in an Azure Storage Account.
    """

    def __init__(self, base_dir=""):
        load_dotenv()
        conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        self.base_dir = base_dir
        self.features_filename = "features.pt"
        self.accuracies_filename = "accuracies.pt"

        self.service = ShareServiceClient.from_connection_string(conn_string)
        self.share = self.service.get_share_client(AZURE_FILESHARE_NAME)

    def _get_file_client(self, relative_path: str):
        """
        Given a relative path to a file from self.base_dir,
        ensures all subdirectories exist and return its directory and file client.
        """

        parts = os.path.join(self.base_dir, relative_path).split("/")
        print("parts", parts)
        current_directory = self.share.get_directory_client("")

        for part in parts[:-1]:
            print("part", part)
            current_directory = current_directory.get_subdirectory_client(part)

            try:
                current_directory.create_directory()
            except ResourceExistsError:
                pass

        file_client = current_directory.get_file_client(parts[-1])
        return file_client

    def _copy_file_to_azure(self, from_path: str, relative_to_path: str):
        """
        Upload a file to the specified path (relative to the client's base path) in azure files.

        Args:
          - from_path (str): The local file path to upload the file from.
          - relative_to_path (str): The file path in the azure storage account to upload the file to,
                (relative to the client's base path)
        """

        file_client = self._get_file_client(relative_to_path)

        with open(from_path, "rb") as data:
            file_client.upload_file(data)

    def _save_torch_object(self, obj: object, relative_to_path: str):
        """
        Upload a torch tensor to the specified path in the azure storage account.
        Replaces the file with the newly saved pytorch object.
        """

        file_client = self._get_file_client(relative_to_path)

        print("saving to", file_client.file_path)

        with io.BytesIO() as data:
            torch.save(obj, data)
            data.seek(0)
            file_client.upload_file(data)

    async def _async_save_torch_object(self, obj: object, relative_to_path: str):
        self._save_torch_object(obj, relative_to_path)

    def _fetch_pt_file(self, relative_from_path: str):
        """
        Download a .pt file from the specified path as a pytorch object and return it.
        From_path must be a path to a .pt file in the azure storage account

        Args:
            from_path (str): The path to the file to download, relative to the base directory.

        Returns:
            The pytorch object stored in the file.
        """

        file_client = self._get_file_client(relative_from_path)
        file_bytes = file_client.download_file().readall()
        file_bytes = io.BytesIO(file_bytes)
        return torch.load(file_bytes)

    def upload_dataset(
        self, features: HPOFeatures, accuracies: List[float], append=False
    ):
        """
        Uploads the dataset consisting of features and accuracies to the specified parent directory.

        Args:
            features (HPOFeatures): The features to be uploaded.
            accuracies (List[float]): The accuracies to be uploaded.
            append (bool, optional): If True, the new data will be appended to the existing data.
                Otherwise, the existing data will be overwritten.

        Returns:
            None
        """

        self._save_torch_object(features, self.features_filename)
        self._save_torch_object(accuracies, self.accuracies_filename)

    async def aupload_dataset(self, features: HPOFeatures, accuracies: List[float]):
        self.upload_dataset(features, accuracies)

    def fetch_dataset(self) -> HPODataset:
        """
        Returns the HPO dataset stored in the base directory.
        Defaults to an empty dataset if no data is found.
        """

        try:
            features = [
                HPOFeatures(*feats)
                for feats in self._fetch_pt_file(self.features_filename)
            ]
            accuracies = self._fetch_pt_file(self.accuracies_filename)
        except ResourceNotFoundError:
            features, accuracies = [], []

        return (features, accuracies)

    def _delete_file(self, relative_file_path: str):
        """
        deletes the given file. if it doesn't exist, does nothing
        """

        file_client = self._get_file_client(relative_file_path)

        try:
            print("to delete: ", file_client.file_path, file_client.file_name)
            file_client.delete_file()
        except ResourceNotFoundError:
            pass

    def _delete_directory(self, relative_dir_path: str = ""):
        """
        deletes the given directory. if it doesn't exist or is nonempty, does nothing
        """

        dir_client = self.share.get_directory_client(
            os.path.join(self.base_dir, relative_dir_path)
        )

        try:
            dir_client.delete_directory()
        except ResourceNotFoundError:
            pass

    def delete_dataset(self):
        """
        deletes the accuracies and features files in the base directory
        if the directory is empty, it will also be deleted
        """

        for filename in [self.features_filename, self.accuracies_filename]:
            self._delete_file(filename)

        self._delete_directory()
