"""
data_ingestion.py
-----------------
Loads the heart disease dataset from a local file or URL,
performs basic inspection, and saves it into the processed directory.
"""

import os
import pandas as pd

# Define paths
RAW_DATA_PATH = "data/raw/heart.csv"
PROCESSED_DATA_PATH = "data/processed/heart_processed.csv"

def load_data(file_path=RAW_DATA_PATH):
    """Load dataset from the given path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please add it to data/raw/.")
    data = pd.read_csv(file_path)
    print(f"✅ Data loaded successfully. Shape: {data.shape}")
    return data

def inspect_data(data: pd.DataFrame):
    """Print basic information about the dataset."""
    print("📊 First 5 rows:")
    print(data.head())
    print("🔍 Data Info:")
    print(data.info())
    print("🧮 Missing Values:")
    print(data.isnull().sum())

def save_processed_data(data: pd.DataFrame, output_path=PROCESSED_DATA_PATH):
    """Save clean data to processed directory."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"💾 Processed data saved at: {output_path}")

def main():
    data = load_data()
    inspect_data(data)
    save_processed_data(data)

if __name__ == "__main__":
    main()