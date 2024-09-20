from azure.core.exceptions import ResourceExistsError
from azure.storage.fileshare import ShareServiceClient
import torch

from dotenv import load_dotenv

import os
import io

load_dotenv()

conn_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

def upload_file(from_path, to_path):

    service = ShareServiceClient.from_connection_string(conn_string)
    share = service.get_share_client('data')

    # Split the directory path into parts (e.g., "some/parent/directory" -> ["some", "parent", "directory"])
    parts = to_path.split('/')
    
    current_directory = share.get_directory_client('')

    for part in parts[:-1]:
        # Move to the next sub-directory
        current_directory = current_directory.get_subdirectory_client(part)
        
        try:
            # Try creating the sub-directory if it doesn't exist
            current_directory.create_directory()

        except ResourceExistsError:
            pass

    file_client = current_directory.get_file_client(parts[-1])

    with open(from_path, 'rb') as data:
        file_client.upload_file(data)


def load_pt_file(from_path):
    """
    download a .pt file from the specified path and convert it to a pytorch object, returns it

    from_path must be a path to a .pt file in the azure storage account
    """

    service = ShareServiceClient.from_connection_string(conn_string)
    share = service.get_share_client('data')
    file_client = share.get_file_client(from_path)
    
    file_bytes = file_client.download_file().readall()
    file_bytes = io.BytesIO(file_bytes)
    return torch.load(file_bytes)


def upload_torch_tensor(tensor, to_path):
    """
    Upload a torch tensor to the specified path in the azure storage account
    """

    service = ShareServiceClient.from_connection_string(conn_string)
    share = service.get_share_client('data')

    # Split the directory path into parts (e.g., "some/parent/directory" -> ["some", "parent", "directory"])
    parts = to_path.split('/')
    
    current_directory = share.get_directory_client('')

    for part in parts[:-1]:
        # Move to the next sub-directory
        current_directory = current_directory.get_subdirectory_client(part)
        
        try:
            # Try creating the sub-directory if it doesn't exist
            current_directory.create_directory()

        except ResourceExistsError:
            pass

    file_client = current_directory.get_file_client(parts[-1])

    with io.BytesIO() as data:
        torch.save(tensor, data)
        data.seek(0)
        file_client.upload_file(data)
