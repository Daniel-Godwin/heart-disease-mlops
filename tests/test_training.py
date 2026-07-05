"""Unit tests for the training data contract and model quality gate."""

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils import TRAIN_PATH, TEST_PATH, TARGET_COL, load_model, load_features


@pytest.fixture(scope="module")
def train_df():
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_PATH)


def test_train_test_schema_match(train_df, test_df):
    assert list(train_df.columns) == list(test_df.columns)
    assert TARGET_COL in train_df.columns


def test_no_missing_values(train_df, test_df):
    assert train_df.isna().sum().sum() == 0
    assert test_df.isna().sum().sum() == 0


def test_target_is_binary(train_df):
    assert set(train_df[TARGET_COL].unique()) <= {0, 1}


def test_features_match_saved_list(train_df):
    features = load_features()
    assert [c for c in train_df.columns if c != TARGET_COL] == features


def test_model_quality_gate(test_df):
    """Deployment gate: model must beat minimum performance on held-out data."""
    model = load_model()
    X = test_df.drop(columns=[TARGET_COL])
    y = test_df[TARGET_COL]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)

    assert acc >= 0.75, f"Accuracy {acc:.3f} below deployment gate (0.75)"
    assert auc >= 0.80, f"ROC-AUC {auc:.3f} below deployment gate (0.80)"
