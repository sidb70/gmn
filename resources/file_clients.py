import os
import errno
import io
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.fileshare import ShareServiceClient
import torch
from typing import Union


load_dotenv()


class FileClient(ABC):
    """
    Abstract class for file clients.

    A FileClient interacts with a file system relative to self.base_dir.
    It shouldn't be able to access or modify files outside of self.base_dir.
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir

    @abstractmethod
    def bytes_to_file(self, data: io.BytesIO, relative_path: str):
        """
        Saves the data to the specified path, relative to self.base_dir.
        If the file already exists, it is overwritten.
        Creates any necessary directories.
        """
        ...

    def str_to_file(self, data: str, relative_path: str):
        """
        Same as bytes_to_file but for strings
        """
        self.bytes_to_file(io.BytesIO(data.encode()), relative_path)

    def obj_to_pt_file(self, obj: object, relative_to_path: str):
        """
        Saves a python object to the specified path using torch.save, relative to self.base_dir.
        If the file already exists, it is overwritten.
        Creates any necessary directories.
        """

        with io.BytesIO() as data:
            torch.save(obj, data)
            data.seek(0)
            self.bytes_to_file(data, relative_to_path)

    async def async_save_torch_object(self, obj: object, relative_to_path: str):
        self.obj_to_pt_file(obj, relative_to_path)

    @abstractmethod
    def read_file_b(self, relative_path: str) -> bytes:
        """
        Reads the file at the specified path, relative to self.base_dir, and returns its contents as bytes.
        If the file doesn't exist, raises FileNotFoundError.
        """
        ...

    @abstractmethod
    def list_directories(self, relative_dir_path: str):
        """
        Returns a list of subdirectories in the directory, relative to self.base_dir.
        """
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
        Deletes the directory, relative to self.base_dir.
        If it is not empty, recursively deletes all its contents.
        If it doesn't exist does nothing.
        """
        ...

    def read_pt_file(self, relative_from_path: str):
        file_bytes = self.read_file_b(relative_from_path)
        with io.BytesIO(file_bytes) as data:
            return torch.load(data)

    def copy_from_local_file(self, from_path: str, relative_to_path: str):
        with open(from_path, "rb") as data:
            self.bytes_to_file(data, relative_to_path)


AZURE_FILESHARE_NAME = "data"


class AzureFileClient(FileClient):

    def __init__(self, base_dir):
        super().__init__(base_dir)
        conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        self.service = ShareServiceClient.from_connection_string(conn_string)
        self.share = self.service.get_share_client(AZURE_FILESHARE_NAME)

    def bytes_to_file(self, data: Union[bytes , io.BytesIO], relative_path: str):
        file_client = self._get_file_client(relative_path)
        file_client.upload_file(data)

    def read_file_b(self, relative_path: str) -> bytes:
        file_client = self._get_file_client(relative_path)
        file_bytes = file_client.download_file().readall()
        return file_bytes
    
    def list_directories(self, relative_dir_path: str):
        dir_client = self.share.get_directory_client(
            os.path.join(self.base_dir, relative_dir_path)
        )

        return [item.name for item in dir_client.list_directories_and_files()]


    def delete_file(self, relative_path: str):
        file_client = self._get_file_client(relative_path)
        try:
            file_client.delete_file()
        except ResourceNotFoundError:
            pass

    def _recursively_delete_directory(self, dir_client):
        for item in dir_client.list_directories_and_files():
            if item.is_directory:
                self._recursively_delete_directory(
                    dir_client.get_subdirectory_client(item.name)
                )
            else:
                dir_client.get_file_client(item.name).delete_file()
        dir_client.delete_directory()

    def delete_directory(self, relative_dir_path: str = ""):

        dir_client = self.share.get_directory_client(
            os.path.join(self.base_dir, relative_dir_path)
        )

        try:
            dir_client.delete_directory()
        except ResourceNotFoundError:
            pass
        except ResourceExistsError:
            # recursively delete all files in the directory
            self._recursively_delete_directory(dir_client)

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

    def __init__(self, base_dir):
        super().__init__(base_dir)

    def bytes_to_file(self, data: io.BytesIO, relative_path: str):
        os.makedirs(
            os.path.join(self.base_dir, os.path.dirname(relative_path)), exist_ok=True
        )
        with open(os.path.join(self.base_dir, relative_path), "wb") as f:
            f.write(data.read())

    def read_file_b(self, relative_path: str) -> bytes:
        with open(os.path.join(self.base_dir, relative_path), "rb") as f:
            return f.read()
        
    def list_directories(self, relative_path: str=""):
        return [dirname for dirname in os.listdir(os.path.join(self.base_dir, relative_path)) \
                if os.path.isdir(os.path.join(self.base_dir, relative_path, dirname))]

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

        except OSError as e:

            if e.errno == errno.ENOENT: # directory doesn't exist
                pass

            elif e.errno == errno.ENOTEMPTY: # directory is not empty
                for root, dirs, files in os.walk(dir_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                os.rmdir(dir_path)
