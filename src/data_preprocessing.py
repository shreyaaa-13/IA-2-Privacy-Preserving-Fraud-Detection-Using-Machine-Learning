"""
Data Preprocessing and Pipeline Management Module.
Handles dataset loading, exploratory diagnostics, leak-free scaling,
chronological vs. stratified data splitting, and metadata logging.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from configs.config import (
    REAL_DATASET_PATH,
    SYNTHETIC_DATASET_PATH,
    RESULTS_DIR,
    RANDOM_SEED,
    TEST_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN
)
from src.synthetic_data import generate_synthetic_fraud_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    """
    Load dataset with automatic fallback hierarchy:
    1. Specified path if provided and exists.
    2. Real ULB Kaggle dataset (`data/creditcard.csv`) if present.
    3. Synthetic verification dataset (`data/synthetic_creditcard.csv`), auto-generated if absent.

    Returns:
    --------
    df : pd.DataFrame
    provenance_tag : str ('REAL-KAGGLE-DATASET' or 'SYNTHETIC-VERIFICATION-DATASET')
    """
    if dataset_path and Path(dataset_path).exists():
        logger.info(f"Loading dataset from user-specified path: {dataset_path}")
        df = pd.read_csv(dataset_path)
        tag = "REAL-KAGGLE-DATASET" if "creditcard.csv" in str(dataset_path) else "CUSTOM-DATASET"
        return df, tag

    if REAL_DATASET_PATH.exists():
        logger.info(f"Detected genuine Kaggle dataset at: {REAL_DATASET_PATH}")
        df = pd.read_csv(REAL_DATASET_PATH)
        return df, "REAL-KAGGLE-DATASET"

    logger.warning(
        f"Real dataset not found at '{REAL_DATASET_PATH}'. "
        "Falling back to synthetic dataset for software pipeline verification. "
        "Place genuine 'creditcard.csv' into 'data/' folder to reproduce benchmark evaluation."
    )
    if not SYNTHETIC_DATASET_PATH.exists():
        logger.info("Generating synthetic verification dataset...")
        generate_synthetic_fraud_data(n_samples=5000, save_path=SYNTHETIC_DATASET_PATH)

    df = pd.read_csv(SYNTHETIC_DATASET_PATH)
    return df, "SYNTHETIC-VERIFICATION-DATASET"


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform rigorous diagnostic inspection of the dataset.

    Inspects:
    - Dimensions (rows, columns)
    - Missing values count per column
    - Duplicate records count
    - Data types
    - Target class distribution and imbalance ratio
    - Memory usage
    """
    total_records = len(df)
    total_columns = df.shape[1]
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    fraud_count = int(class_counts.get(1, 0))
    legit_count = int(class_counts.get(0, 0))
    fraud_ratio = float(fraud_count / total_records) if total_records > 0 else 0.0

    inspection_summary = {
        "total_records": total_records,
        "total_columns": total_columns,
        "column_names": list(df.columns),
        "total_missing_values": total_missing,
        "missing_per_column": {k: int(v) for k, v in missing_counts.items() if v > 0},
        "duplicate_records": duplicate_count,
        "class_distribution": {
            "legitimate_0": legit_count,
            "fraudulent_1": fraud_count,
            "fraud_prevalence_percentage": round(fraud_ratio * 100, 4)
        },
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

    logger.info(
        f"Dataset Inspection Complete: {total_records} rows, {total_columns} cols, "
        f"{fraud_count} fraud ({inspection_summary['class_distribution']['fraud_prevalence_percentage']}%), "
        f"{duplicate_count} duplicates, {total_missing} missing values."
    )
    return inspection_summary


def split_data(
    df: pd.DataFrame,
    method: str = "stratified",
    test_size: float = TEST_SPLIT_RATIO,
    val_size: float = VAL_SPLIT_RATIO,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Split dataset into train, validation, and test splits while strictly preventing leakage.

    Supports two primary split methodologies:
    1. 'stratified' (Default & Recommended for Imbalanced ML):
       Maintains exact class proportions across splits, ensuring both train and test
       contain sufficient fraudulent transactions for statistically meaningful evaluation.
    2. 'chronological':
       Sorts strictly by `Time` column. Simulates production deployment where models
       are trained on historical transactions and evaluated on future transactions.
       Limitations: Vulnerable to temporal non-stationarity, concept drift, and uneven
       distribution of rare fraud events across arbitrary time boundaries.

    Returns:
    --------
    train_df, test_df, val_df, split_metadata
    """
    total_len = len(df)

    if method == "chronological":
        logger.info("Executing chronological split based on 'Time' column...")
        df_sorted = df.sort_values(by=TIME_COLUMN).reset_index(drop=True)

        test_idx = int(total_len * (1.0 - test_size))
        val_idx = int(test_idx - (total_len * val_size))

        train_df = df_sorted.iloc[:val_idx].copy()
        val_df = df_sorted.iloc[val_idx:test_idx].copy()
        test_df = df_sorted.iloc[test_idx:].copy()

        split_notes = (
            "Chronological split applied on 'Time'. Real-world simulation limitation: "
            "temporal shift and non-uniform fraud incidence between training and evaluation windows."
        )

    elif method == "stratified":
        logger.info("Executing stratified split to preserve extreme class imbalance...")
        # First split into train_val and test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[TARGET_COLUMN],
            random_state=random_state
        )

        # Then split train_val into train and val
        adjusted_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            stratify=train_val_df[TARGET_COLUMN],
            random_state=random_state
        )

        split_notes = (
            "Stratified split applied. Class distribution strictly preserved across "
            "train, validation, and test partitions."
        )
    else:
        raise ValueError(f"Unknown split method: {method}. Choose 'stratified' or 'chronological'.")

    metadata = {
        "method": method,
        "split_notes": split_notes,
        "train_size": len(train_df),
        "val_size": len(val_df) if val_df is not None else 0,
        "test_size": len(test_df),
        "train_fraud_count": int(train_df[TARGET_COLUMN].sum()),
        "test_fraud_count": int(test_df[TARGET_COLUMN].sum()),
        "train_fraud_rate": float(train_df[TARGET_COLUMN].mean()),
        "test_fraud_rate": float(test_df[TARGET_COLUMN].mean())
    }

    logger.info(
        f"Split complete [{method}]: Train={len(train_df)} ({metadata['train_fraud_count']} fraud), "
        f"Val={len(val_df)}, Test={len(test_df)} ({metadata['test_fraud_count']} fraud)."
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), val_df.reset_index(drop=True), metadata


def preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    drop_time: bool = True,
    scale_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray], list, Any]:
    """
    Separate features and target, and apply leak-free scaling.

    Crucial Anti-Leakage Protocol:
    The scaler (`RobustScaler`) is fitted EXCLUSIVELY on the training features.
    Validation and test sets are transformed using the training set's computed statistics (median, IQR).
    No test distribution statistics are accessed during fitting.

    Features V1 through V28 are already PCA components (standardized).
    Only `Amount` (and `Time` if kept) requires robust scaling due to extreme outlier amounts.

    Returns:
    --------
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names, scaler
    """
    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    if drop_time and TIME_COLUMN in feature_cols:
        feature_cols.remove(TIME_COLUMN)

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COLUMN].values.astype(int)

    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COLUMN].values.astype(int)

    X_val = val_df[feature_cols].copy() if val_df is not None else None
    y_val = val_df[TARGET_COLUMN].values.astype(int) if val_df is not None else None

    scaler = None
    if scale_features:
        # RobustScaler is resilient to extreme transaction amounts
        scaler = RobustScaler()
        # Fit ONLY on training data
        cols_to_scale = [col for col in [AMOUNT_COLUMN] if col in feature_cols]
        if cols_to_scale:
            X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
            X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
            if X_val is not None:
                X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    return (
        X_train.values,
        X_test.values,
        X_val.values if X_val is not None else None,
        y_train,
        y_test,
        y_val,
        feature_cols,
        scaler
    )


def save_preprocessing_metadata(metadata: Dict[str, Any], filepath: Path = RESULTS_DIR / "preprocessing_metadata.json"):
    """Persist preprocessing and split configuration metadata."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Preprocessing metadata persisted to: {filepath}")


if __name__ == "__main__":
    df, tag = load_dataset()
    insp = inspect_dataset(df)
    train_df, test_df, val_df, split_meta = split_data(df, method="stratified")
    X_tr, X_te, X_v, y_tr, y_te, y_v, feats, scaler = preprocess_features(train_df, test_df, val_df)
    meta = {
        "provenance_tag": tag,
        "inspection": insp,
        "split_metadata": split_meta,
        "feature_count": len(feats),
        "features": feats
    }
    save_preprocessing_metadata(meta)
    print("Preprocessed shapes:", X_tr.shape, X_te.shape)
