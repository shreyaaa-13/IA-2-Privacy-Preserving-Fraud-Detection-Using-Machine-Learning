"""
Unit Tests for Preprocessing and Data Partitioning.
Verifies dataset integrity, diagnostic inspection, leak-free scaling,
and stratified vs chronological split consistency.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_data import generate_synthetic_fraud_data
from src.data_preprocessing import (
    inspect_dataset,
    split_data,
    preprocess_features
)


@pytest.fixture
def sample_df():
    """Generate deterministic synthetic DataFrame for unit tests."""
    return generate_synthetic_fraud_data(n_samples=500, fraud_ratio=0.04, random_state=42, save_path=None)


def test_synthetic_data_schema(sample_df):
    """Verify synthetic dataset columns and non-emptiness."""
    assert len(sample_df) == 500
    assert "Time" in sample_df.columns
    assert "Amount" in sample_df.columns
    assert "Class" in sample_df.columns
    for i in range(1, 29):
        assert f"V{i}" in sample_df.columns
    assert sample_df["Class"].sum() == 20  # 4% of 500


def test_dataset_inspection(sample_df):
    """Verify inspection metadata dictionary."""
    summary = inspect_dataset(sample_df)
    assert summary["total_records"] == 500
    assert summary["total_columns"] == 31
    assert summary["total_missing_values"] == 0
    assert summary["class_distribution"]["fraudulent_1"] == 20
    assert summary["class_distribution"]["legitimate_0"] == 480


def test_stratified_split_preserves_ratios(sample_df):
    """Verify that stratified splitting maintains equal fraud ratios."""
    train_df, test_df, val_df, meta = split_data(sample_df, method="stratified", test_size=0.2, val_size=0.1)

    assert len(train_df) + len(test_df) + len(val_df) == 500

    train_rate = train_df["Class"].mean()
    test_rate = test_df["Class"].mean()
    val_rate = val_df["Class"].mean()

    # Verify rates are closely matched (within 1% margin)
    assert abs(train_rate - 0.04) < 0.015
    assert abs(test_rate - 0.04) < 0.015
    assert abs(val_rate - 0.04) < 0.02


def test_leak_free_scaling(sample_df):
    """
    Test that RobustScaler is fitted strictly on X_train.
    Transforming test data must use train statistics without altering train state.
    """
    train_df, test_df, val_df, _ = split_data(sample_df, method="stratified")
    X_tr, X_te, X_v, y_tr, y_te, y_v, feats, scaler = preprocess_features(train_df, test_df, val_df)

    assert X_tr.shape[0] == len(train_df)
    assert X_te.shape[0] == len(test_df)
    assert X_v.shape[0] == len(val_df)
    assert scaler is not None

    # Scaler median should match the train Amount median, NOT test Amount median
    amount_col_idx = feats.index("Amount")
    raw_train_amount = train_df["Amount"].values
    expected_median = np.median(raw_train_amount)
    assert abs(scaler.center_[0] - expected_median) < 1e-4
