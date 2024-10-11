import os
import json
from typing import List
from azure.core.exceptions import ResourceNotFoundError
from preprocessing.preprocessing_types import TrainedNNResult, HPODataset, HPOFeatures
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

        self.features_filename = "model_features.pt"

    def save_model_result(self, result: TrainedNNResult):

        model_dir = str(result.model_id)

        self.file_client.delete_directory(model_dir)

        for i, epoch_feats in enumerate(result.epoch_feats):
            # save epoch feats individually
            self.file_client.obj_to_pt_file(
                epoch_feats, os.path.join(model_dir, f"epoch_{i}_feats.pt")
            )

        results = {
            "hyperparameters": result.hpo_vec,
            "torch_model": result.torch_model,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "accuracy": result.final_accuracy,
        }

        self.file_client.str_to_file(
            json.dumps(results), os.path.join(model_dir, "results.json")
        )

    def save_dataset(self, model_results: list[TrainedNNResult]):
        """
        Saves results for all models.
        """
        for result in model_results:
            self.save_model_result(result)

    async def asave_dataset(self):
        raise NotImplementedError

    def read_dataset(self) -> HPODataset:
        """
        Reads the dataset saved at self.base_dir.

        Returns:
        (per model:)
        - List of feats (per epoch)
        - List of train and val losses (per epoch)
        - hyperparams
        - final accuracy
        """
        model_dirs = self.file_client.list_directories()

        features: List[HPOFeatures] = []
        labels: List[float] = []

        for model_dir in model_dirs:

            epoch0_feats = self.file_client.read_pt_file(
                os.path.join(model_dir, "epoch_0_feats.pt")
            )

            results = json.loads(
                self.file_client.read_file_b(os.path.join(model_dir, "results.json"))
            )

            # print("Results: ", results) 
            # for i, epoch_feats in enumerate(model_features):
            #     # save epoch feats individually
            #     self.file_client.obj_to_pt_file(
            #         epoch_feats, os.path.join(model_dir, f"epoch_{i}_feats.pt")
            #     )
            
            # epoch0_feats=model_features[0],
            train_losses=results["train_losses"]
            val_losses=results["val_losses"]
            final_accuracy=results["accuracy"]
            hpo_vec=results["hyperparameters"]
            features.append([epoch0_feats, hpo_vec])
            labels.append(val_losses[-1])



        return features, labels