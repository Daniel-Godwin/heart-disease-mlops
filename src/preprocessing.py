"""
preprocessing.py
----------------
Handles splitting, encoding, scaling, and saving train-test data
for production ML pipeline (heart disease prediction).

Clean + encode + scale + split dataset properly (NO DATA LEAKAGE FIXED)
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

INPUT_PATH = "data/processed/heart_processed.csv"
OUTPUT_DIR = "data/processed/"

TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")

def load_data(path=INPUT_PATH):
    data = pd.read_csv(path)
    print(f"Loaded data: {data.shape}")
    return data


def clean_data(data: pd.DataFrame):
    # IMPORTANT FIX: remove NaN in target first
    if "target" not in data.columns:
        raise ValueError("Target column 'target' not found!")

    data = data.dropna(subset=["target"])  # FIXED LINE

    # drop remaining NaNs
    data = data.dropna()

    print(f"After cleaning: {data.shape}")
    return data


def encode_features(data: pd.DataFrame):
    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    return data


def split_data(data: pd.DataFrame, target_column="target"):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    print("Target unique values:", y.unique())
    print("Missing in y:", y.isna().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ✅ PRESERVE COLUMN NAMES
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # ✅ PERSIST SCALER for inference (API + predict.py depend on it)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
    print("✅ Scaler saved to models/scaler.joblib")

    return X_train_scaled, X_test_scaled, scaler


def save_data(X_train, X_test, y_train, y_test):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ✅ Reset index BEFORE concat (correct alignment)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("✅ Saved train/test data")
    print("Train columns:", train_df.columns.tolist())
        


def main():
    data = load_data()
    data = clean_data(data)
    data = encode_features(data)

    X_train, X_test, y_train, y_test = split_data(data)

    X_train, X_test, scaler = scale_features(X_train, X_test)
    print("Features after scaling:", X_train.columns.tolist())

    save_data(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()