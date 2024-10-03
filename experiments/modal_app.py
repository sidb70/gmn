import modal
import modal.gpu
import time
from experiments.run_train_hpo import load_data
from preprocessing.generate_data import train_random_cnns_hyperparams
from preprocessing.preprocessing_types import RandHyperparamsConfig, RandCNNConfig
from train.train_hpo2 import train_hpo_mpnn
from train.utils import split
from experiments.train_1_cnn import train_1_cnn


app = modal.App("hpo")
vol = modal.Volume.from_name("hpo-volume")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .micromamba(python_version="3.12")
    .micromamba_install(
        "cupy",
        "pkg-config",
        "libjpeg-turbo",
        "opencv",
        "pytorch",
        "torchvision",
        "cudatoolkit=11.3",
        "numba",
        channels=["pytorch", "conda-forge"],
    )  # ffcv setup: https://github.com/libffcv/ffcv
    .apt_install("build-essential")
    .pip_install_from_requirements("requirements.txt")
)


@app.function(
    gpu=modal.gpu.A10G(), image=image, retries=0, volumes={"/data": vol}, timeout=3600
)
def train_one_cnn():
    train_1_cnn()


@app.function(
    gpu=modal.gpu.A10G(count=4),
    image=image,
    retries=0,
    volumes={"/data": vol},
    timeout=3600,
)
def generate_data():

    import subprocess

    print("here's my gpu:")
    subprocess.run(["nvidia-smi", "--list-gpus"], check=True)

    start_time = time.time()
    n_architectures = 15
    train_random_cnns_hyperparams(
        "../data/hpo",
        n_architectures=n_architectures,
        random_cnn_config=RandCNNConfig(),
        random_hyperparams_config=RandHyperparamsConfig(
            momentum_range=[0.5, 0.5],  # doesn't matter bc using adamw
            n_epochs_range=[50, 51],
        ),
    )
    print(f"Trained {n_architectures} CNNs in {time.time() - start_time:.2f} seconds.")


@app.function(gpu="A10G", image=image, retries=0, volumes={"/data": vol})
def train_hpo_model():
    feats, accuracies = load_data("../data/hpo")

    valid_size = 0.1
    test_size = 0.1
    train_set, valid_set, test_set = split(feats, accuracies, test_size, valid_size)

    feats_train, labels_train = train_set
    feats_valid, labels_valid = valid_set
    feats_test, labels_test = test_set

    mpnn = train_hpo_mpnn(feats_train, labels_train)


@app.local_entrypoint()
def main():
    train_one_cnn.remote()
