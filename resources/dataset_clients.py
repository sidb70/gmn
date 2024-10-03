import os
import io
from dotenv import load_dotenv

from typing import List
from preprocessing.preprocessing_types import HPOFeatures, HPODataset
from .file_clients import FileClient, LocalFileClient

from abc import ABC, abstractmethod


AZURE_FILESHARE_NAME = "data"


class HPODatasetClient:
    """
    Abstract class for managing the storage of an HPO dataset in the format:

    base_dir/
        model_id
        epoch0.pt
        ...
        final_model.pt
        results.json: {
            "train_losses": [float],
            "val_losses": [float],
            "final_accuracy": float
            "hyperparameters": hpo_vec
        }
    """

    def __init__(self, file_storage_client: FileClient = None, base_dir=""):

        if file_storage_client is None:
            self.file_storage_client = LocalFileClient(base_dir)

        self.features_filename = "features.pt"
        self.accuracies_filename = "accuracies.pt"

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

        self.save_torch_object(features, self.features_filename)
        self.save_torch_object(accuracies, self.accuracies_filename)

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
                for feats in self.read_pt_file(self.features_filename)
            ]
            accuracies = self.read_pt_file(self.accuracies_filename)
        except ResourceNotFoundError:
            features, accuracies = [], []

        return (features, accuracies)

    def delete_dataset(self):
        """
        deletes the accuracies and features files in the base directory
        if the directory is empty, it will also be deleted
        """

        for filename in [self.features_filename, self.accuracies_filename]:
            self.delete_file(filename)

        self._delete_directory()
