import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import unittest
import shutil
import torch
from resources import LocalFileClient, AzureFileClient, FileClient


class TestTrainSaveLocally(unittest.TestCase):

    # @unittest.skip("skip")
    def test_save_load_tensor(self):

        file_clients: list[FileClient] = [
            LocalFileClient("data/test/file-client"),
            AzureFileClient("test/file-client"),
        ]

        for file_client in file_clients:
            file_client.delete_directory()
            tensors = [torch.rand(4, 3)]
            json_data = {"tensor": tensors}

            file_client.obj_to_pt_file(tensors, "subdir/test_tensor.pt")
            file_client.obj_to_pt_file(json_data, "subdir/test_json.pt")

            read_tensors = file_client.read_pt_file("subdir/test_tensor.pt")
            read_json = file_client.read_pt_file("subdir/test_json.pt")

            self.assertTrue(torch.equal(tensors[0], read_tensors[0]))
            self.assertTrue(torch.equal(json_data["tensor"][0], read_json["tensor"][0]))

            file_client.delete_directory("nonexistent")
            file_client.delete_directory()

    # @unittest.skip("skip")
    def test_read_copy_subdirectories(self):

        file_clients: list[FileClient] = [
            LocalFileClient("data/test/file-client"),
            AzureFileClient("test/file-client"),
        ]

        for file_client in file_clients:

            shutil.rmtree("data/test/local-dir", ignore_errors=True)
            file_client.delete_directory()

            os.makedirs("data/test/local-dir", exist_ok=True)
            with open("data/test/local-dir/a.pt", "wb") as f:
                torch.save(torch.rand(2), f)

            os.makedirs("data/test/local-dir/subdir", exist_ok=True)
            with open("data/test/local-dir/subdir/b.pt", "wb") as f:
                torch.save(torch.rand(2), f)

            os.makedirs("data/test/local-dir/emptysubdir", exist_ok=True)

            file_client.copy_dir_from_local("data/test/local-dir", "client_dir")

            # Test whether the files were copied from local filesystem to the file client's filesystem
            client_dir_subdirs = file_client.list_directories("client_dir")

            # emptydir should not be copied, and the file a.pt shouldn't appear in the list of directories
            self.assertEqual(client_dir_subdirs, ["subdir"])

            os.makedirs("data/test/local-dir/subdir2/subdir3", exist_ok=True)
            with open("data/test/local-dir/subdir2/subdir3/c.pt", "wb") as f:
                torch.save(torch.rand(2), f)

            file_client.copy_dir_from_local("data/test/local-dir", "client_dir")

            # Test whether the files were copied from local filesystem to the file client's filesystem
            client_dir_subdirs = file_client.list_directories("client_dir")

            # should inclued both subdir and subdir2 now
            self.assertEqual(client_dir_subdirs, ["subdir", "subdir2"])

            # check whether the subdirectories were copied
            subdir2_subdirs = file_client.list_directories("client_dir/subdir2")
            self.assertEqual(subdir2_subdirs, ["subdir3"])

            file_client.copy_file_from_local("data/test/local-dir/a.pt", "A.pt")
            a = file_client.read_pt_file("A.pt")
            self.assertTrue(
                torch.equal(
                    torch.load("data/test/local-dir/a.pt", weights_only=False), a
                )
            )

            file_client.delete_directory()

    def test_copy_dir_overwrite_file(self):

        file_clients: list[FileClient] = [
            LocalFileClient("data/test/file-client-3"),
            AzureFileClient("test/file-client-3"),
        ]

        for file_client in file_clients:
            file_client.delete_directory()

            os.makedirs("data/test/local-dir", exist_ok=True)
            with open("data/test/local-dir/a.pt", "wb") as f:
                torch.save(torch.tensor(2), f)

            file_client.copy_dir_from_local("data/test/local-dir", "client_dir")
            a = file_client.read_pt_file("client_dir/a.pt")
            self.assertTrue(torch.equal(torch.tensor(2), a))

            # modify the file locally
            with open("data/test/local-dir/a.pt", "wb") as f:
                torch.save(torch.tensor(3), f)

            # copy its parent directory again
            file_client.copy_dir_from_local("data/test/local-dir", "client_dir")
            a = file_client.read_pt_file("client_dir/a.pt")

            # check whither it was overwritten
            self.assertTrue(torch.equal(torch.tensor(3), a))


if __name__ == "__main__":
    unittest.main()
