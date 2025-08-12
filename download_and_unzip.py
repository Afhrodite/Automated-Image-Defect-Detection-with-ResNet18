import os
import urllib.request
import zipfile


# Config
POSITIVE_URL = (
    "https://s3-api.us-geo.objectstorage.softlayer.net/"
    "cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip"
)
NEGATIVE_URL = (
    "https://s3-api.us-geo.objectstorage.softlayer.net/"
    "cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip"
)

POSITIVE_FILE = "Positive_tensors.zip"
NEGATIVE_FILE = "Negative_tensors.zip"


# Functions
def download_file(url, filename):
    """
    Download a file from a given URL if it does not already exist.
    """
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists. Skipping download.")


def unzip_file(zip_path, extract_to="."):
    """
    Unzip a .zip file to the specified directory.
    """
    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path}")


def main():
    """
    Download and unzip both Positive and Negative tensors.
    """
    # Positive tensors
    download_file(POSITIVE_URL, POSITIVE_FILE)
    unzip_file(POSITIVE_FILE)

    # Negative tensors
    download_file(NEGATIVE_URL, NEGATIVE_FILE)
    unzip_file(NEGATIVE_FILE)


if __name__ == "__main__":
    main()