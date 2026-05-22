from pathlib import Path
from zipfile import ZipFile
from sklearn.model_selection import train_test_split

import pandas as pd
import requests

DATA_URL = "http://alt.qcri.org/semeval2016/task6/data/uploads/stancedataset.zip"
DATA_PATH = Path(__file__).parents[1] / "data" / "processed" / "stance_dataset.csv"


def download_and_extract(url: str):
    extract_to = Path(__file__).parents[1] / "data" / "raw"
    extract_to.mkdir(parents=True, exist_ok=True)
    local_filename = url.split("/")[-1]
    local_path = extract_to / local_filename
    if not local_path.exists():
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {local_filename}")
    else:
        print(f"{local_filename} already exists, skipping download.")
    with ZipFile(local_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {local_filename} to {extract_to}")


def preprocess():
    train_path = (
        Path(__file__).parents[1] / "data" / "raw" / "StanceDataset" / "train.csv"
    )
    test_path = (
        Path(__file__).parents[1] / "data" / "raw" / "StanceDataset" / "test.csv"
    )
    train_df = pd.read_csv(train_path, encoding="latin-1", engine="python")
    train_df["Target-Stance"] = train_df["Target"] + "-" + train_df["Stance"]
    train_df, valid_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["Target-Stance"]
    )
    train_df = train_df.drop(columns=["Target-Stance"])
    train_df["Split"] = "train"
    valid_df = valid_df.drop(columns=["Target-Stance"])
    valid_df["Split"] = "valid"
    test_df = pd.read_csv(test_path, encoding="latin-1", engine="python")
    test_df["Split"] = "test"
    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    # columns?
    print("Columns:", df.columns.tolist())
    # check if Tweet column contains duplicates
    duplicate_tweets = df["Tweet"].duplicated().sum()
    print(f"Number of duplicate tweets: {duplicate_tweets}")
    # print statistics
    print("Dataset statistics:")
    for split in ["train", "valid", "test"]:
        split_df = df[df["Split"] == split]
        print(f"{split}: {len(split_df)} samples")
        for target in split_df["Target"].unique():
            target_df = split_df[split_df["Target"] == target]
            stance_counts = target_df["Stance"].value_counts()
            print(f"  {target}: {stance_counts.to_dict()}")
    # save df to data/processed/stance_dataset.csv
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved processed dataset to {DATA_PATH}")


if __name__ == "__main__":
    download_and_extract(DATA_URL)
    preprocess()
