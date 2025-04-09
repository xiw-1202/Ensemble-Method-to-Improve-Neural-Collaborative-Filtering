#!/usr/bin/env python
"""
Script to download the MovieLens dataset
"""
import os
import zipfile
import urllib.request
import shutil

# URLs for MovieLens datasets
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

# Local paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

def download_dataset(url, dataset_name):
    """
    Download and extract dataset
    """
    # Create directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Download zip file
    zip_path = os.path.join(RAW_DATA_DIR, f"{dataset_name}.zip")
    print(f"Downloading {dataset_name} dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract zip file
    print(f"Extracting {dataset_name} dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
    
    # Remove the zip file
    os.remove(zip_path)
    print(f"{dataset_name} dataset downloaded and extracted successfully.")

if __name__ == "__main__":
    # Download the 1M dataset by default, change to MOVIELENS_100K_URL if needed
    download_dataset(MOVIELENS_1M_URL, "ml-1m")
    print("Dataset directory structure:")
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        level = root.replace(RAW_DATA_DIR, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")
