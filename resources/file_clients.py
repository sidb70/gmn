import os
import io
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.fileshare import ShareServiceClient
import torch


load_dotenv()


class FileClient(ABC):

    def __init__(self, base_dir=""):
        self.base_dir = base_dir

    @abstractmethod
    def save_to_file(self, data: io.BytesIO): 
        ...

    @abstractmethod
    def read_file(self, relative_path: str) -> io.BytesIO:
        ...

    @abstractmethod
    def delete_file(self, relative_file_path: str): 
        """
        Deletes the file. If it doesn't exist, does nothing
        """
        ...

    @abstractmethod
    def delete_directory(self, relative_dir_path: str): 
        """
        Deletes the directory. If it doesn't exist or is not empty, does nothing.
        """
        ...


    def save_torch_object(self, obj: object, relative_to_path: str):
        with io.BytesIO() as data:
            torch.save(obj, data)
            data.seek(0)
            self.save_to_file(data, relative_to_path)

    async def async_save_torch_object(self, obj: object, relative_to_path: str):
        self.save_torch_object(obj, relative_to_path)

    def read_pt_file(self, relative_from_path: str):
        file_bytes = self.read_file(relative_from_path)
        return torch.load(file_bytes)

    def copy_from_local_file(self, from_path: str, relative_to_path: str):
        with open(from_path, "rb") as data:
            self.save_to_file(data, relative_to_path)


AZURE_FILESHARE_NAME = "data"

class AzureFileClient(FileClient):

    def __init__(self, base_dir=""):
        super().__init__(base_dir)
        conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        self.service = ShareServiceClient.from_connection_string(conn_string)
        self.share = self.service.get_share_client(AZURE_FILESHARE_NAME)

    def save_to_file(self, data: io.BytesIO, relative_path: str):
        file_client = self._get_file_client(relative_path)
        file_client.upload_file(data)

    def read_file(self, relative_path: str) -> io.BytesIO:
        file_client = self._get_file_client(relative_path)
        file_bytes = file_client.download_file().readall()
        file_bytes = io.BytesIO(file_bytes)
        return file_bytes

    def delete_file(self, relative_path: str):
        file_client = self._get_file_client(relative_path)
        try:
            file_client.delete_file()
        except ResourceNotFoundError:
            pass

    def delete_directory(self, relative_dir_path: str = ""):

        dir_client = self.share.get_directory_client(
            os.path.join(self.base_dir, relative_dir_path)
        )

        try:
            dir_client.delete_directory()
        except ResourceNotFoundError:
            pass

    def _get_file_client(self, relative_path: str):

        parts = os.path.join(self.base_dir, relative_path).split("/")
        current_directory = self.share.get_directory_client("")

        for part in parts[:-1]:
            current_directory = current_directory.get_subdirectory_client(part)
            try:
                current_directory.create_directory()
            except ResourceExistsError:
                pass

        file_client = current_directory.get_file_client(parts[-1])
        return file_client


class LocalFileClient(FileClient):

    def __init__(self, base_dir=""):
        super().__init__(base_dir)

    def save_to_file(self, data: io.BytesIO, relative_path: str):
        with open(os.path.join(self.base_dir, relative_path), "wb") as f:
            f.write(data.read())

    def read_file(self, relative_path: str) -> io.BytesIO:
        with open(os.path.join(self.base_dir, relative_path), "rb") as f:
            return io.BytesIO(f.read())

    def delete_file(self, relative_file_path: str):
        file_path = os.path.join(self.base_dir, relative_file_path)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    def delete_directory(self, relative_dir_path: str = ""):
        dir_path = os.path.join(self.base_dir, relative_dir_path)
        try:
            os.rmdir(dir_path)
        except FileNotFoundError:
            pass
