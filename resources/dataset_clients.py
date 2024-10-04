import os
import json
from azure.core.exceptions import ResourceNotFoundError
from preprocessing.preprocessing_types import TrainedNNResult
from .file_clients import FileClient, LocalFileClient


AZURE_FILESHARE_NAME = "data"


class HPOExperimentClient:
    """
    Abstract class for managing the storage of an HPO dataset in the format:

    base_dir/
        model_id/
            model_features: list of epoch feats
            results.json
    """

    def __init__(self, file_client: FileClient = None):

        if file_client is None:
            file_client = LocalFileClient()
        self.file_client = file_client

    def save_model_result(self, result: TrainedNNResult):

        model_dir = str(result.model_id)

        self.file_client.delete_directory(model_dir)

        self.file_client.obj_to_pt_file(
            result.epoch_feats, os.path.join(model_dir, "model_features.pt")
        )

        results = {
            "hyperparameters": result.hpo_vec,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "accuracy": result.final_accuracy,
        }

        self.file_client.str_to_file(
            json.dumps(results), os.path.join(model_dir, "results.json")
        )

        print(
            "Saved model {} to {}".format(
                result.model_id, os.path.join(self.file_client.base_dir, model_dir)
            )
        )

    def save_dataset(self, model_results: list[TrainedNNResult]):
        """
        Saves results for all models.
        """
        for result in model_results:
            self.save_model_result(result)

    async def asave_dataset(self):
        raise NotImplementedError

    def read_dataset(self) -> list[TrainedNNResult]:
        """
        Reads the dataset saved at self.base_dir.
        """
        raise NotImplementedError



    def delete_dataset(self):
        """
        Deletes the contents of self.base_dir
        """
        self.file_client.delete_directory()
