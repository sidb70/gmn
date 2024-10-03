import os


class LocalDatasetClient:

    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
