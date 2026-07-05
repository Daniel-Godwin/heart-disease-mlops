"""Unit tests for src/preprocessing.py"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean_data, encode_features, split_data, scale_features


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "chol": rng.integers(120, 400, n),
        "sex_label": rng.choice(["male", "female"], n),
        "target": rng.integers(0, 2, n),
    })


def test_clean_data_drops_nan_target(sample_df):
    df = sample_df.copy()
    df.loc[0, "target"] = np.nan
    cleaned = clean_data(df)
    assert cleaned["target"].isna().sum() == 0
    assert len(cleaned) == len(df) - 1


def test_clean_data_requires_target_column(sample_df):
    with pytest.raises(ValueError):
        clean_data(sample_df.drop(columns=["target"]))


def test_encode_features_converts_objects_to_numeric(sample_df):
    encoded = encode_features(sample_df.copy())
    assert encoded.select_dtypes(include=["object"]).empty
    assert pd.api.types.is_numeric_dtype(encoded["sex_label"])


def test_split_data_shapes_and_stratification(sample_df):
    df = encode_features(sample_df.copy())
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) == len(y_train)
    assert "target" not in X_train.columns
    # stratification keeps class balance within a few points
    assert abs(y_train.mean() - y_test.mean()) < 0.15


def test_scale_features_standardizes_train(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # scaler is saved relative to cwd
    df = encode_features(sample_df.copy())
    X_train, X_test, _, _ = split_data(df)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # train features ~ zero mean, unit variance
    assert np.allclose(X_train_s.mean().values, 0, atol=1e-6)
    assert np.allclose(X_train_s.std(ddof=0).values, 1, atol=1e-6)
    # column names preserved
    assert list(X_train_s.columns) == list(X_train.columns)
    # scaler persisted
    assert (tmp_path / "models" / "scaler.joblib").exists()
