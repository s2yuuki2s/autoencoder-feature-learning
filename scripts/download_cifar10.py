import os
import urllib.request
import tarfile

DATA_DIR = os.path.join("data", "cifar-10-batches-bin")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
TAR_PATH = os.path.join("data", "cifar-10-binary.tar.gz")


def download_and_extract():
    if os.path.exists(DATA_DIR):
        print("CIFAR-10 already downloaded.")
        return

    os.makedirs("data", exist_ok=True)

    print("Downloading CIFAR-10...")
    urllib.request.urlretrieve(URL, TAR_PATH)

    print("Extracting...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall("data")

    os.remove(TAR_PATH)
    print("Done.")


if __name__ == "__main__":
    download_and_extract()
