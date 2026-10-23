"""
Synthetic Dataset Generator for Software Verification and Offline Pipeline Testing.

IMPORTANT ACADEMIC INTEGRITY NOTICE:
This synthetic generator produces a small, mock dataset strictly for unit testing,
CI/CD verification, and validating pipeline mechanics before the genuine Kaggle
dataset is supplied by the user.

Per academic integrity standards:
- Synthetic results MUST NOT be represented as real fraud detection benchmarks.
- Any output generated from this synthetic data is explicitly marked with `[SYNTHETIC-TEST]`.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from configs.config import (
    SYNTHETIC_DATASET_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN
)


def generate_synthetic_fraud_data(
    n_samples: int = 5000,
    fraud_ratio: float = 0.005,  # 0.5% fraud rate to simulate severe class imbalance
    random_state: int = RANDOM_SEED,
    save_path: Path = SYNTHETIC_DATASET_PATH
) -> pd.DataFrame:
    """
    Generate a synthetic dataset matching the exact schema of ULB Credit Card Fraud dataset:
    - Time: elapsed seconds
    - V1 through V28: 28 continuous numerical features (simulating PCA components)
    - Amount: transaction amount in USD/EUR
    - Class: binary indicator (0 = legitimate, 1 = fraudulent)

    Parameters:
    -----------
    n_samples : int
        Total number of records to generate (default 5000 for fast verification).
    fraud_ratio : float
        Proportion of fraudulent transactions (default 0.005 = 25 fraud samples).
    random_state : int
        Seed for deterministic generation.
    save_path : Path
        Destination path for saving the CSV.

    Returns:
    --------
    pd.DataFrame:
        DataFrame containing synthetic transactions.
    """
    rng = np.random.RandomState(random_state)
    n_fraud = max(2, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # 1. Simulate Time: sequential timestamps over a 48-hour window (0 to 172,800 seconds)
    time_legit = np.sort(rng.uniform(0, 172800, size=n_legit))
    time_fraud = np.sort(rng.uniform(0, 172800, size=n_fraud))
    time_all = np.concatenate([time_legit, time_fraud])

    # 2. Simulate V1 to V28 (PCA-like features):
    # Legitimate: standard normal distributions
    pca_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    # Fraudulent: shifted distributions with higher variance (features V14, V12, V10 typically differ)
    shift = np.zeros(28)
    shift[11] = -2.5  # V12 shift
    shift[13] = -3.0  # V14 shift
    shift[16] = -2.0  # V17 shift
    pca_fraud = rng.normal(loc=shift, scale=1.8, size=(n_fraud, 28))
    pca_all = np.vstack([pca_legit, pca_fraud])

    # 3. Simulate Amount:
    # Legitimate: log-normal distribution with median around 25-50
    amount_legit = rng.lognormal(mean=3.2, sigma=1.2, size=n_legit)
    # Fraudulent: bimodal (some tiny authorization amounts, some high amounts)
    amount_fraud = np.where(
        rng.rand(n_fraud) > 0.5,
        rng.uniform(1.0, 10.0, size=n_fraud),
        rng.uniform(100.0, 800.0, size=n_fraud)
    )
    amount_all = np.concatenate([amount_legit, amount_fraud])

    # 4. Target Labels
    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # Construct DataFrame
    columns = [TIME_COLUMN] + [f"V{i}" for i in range(1, 29)] + [AMOUNT_COLUMN, TARGET_COLUMN]
    data_matrix = np.column_stack([time_all, pca_all, amount_all, labels])
    df = pd.DataFrame(data_matrix, columns=columns)

    # Cast types properly
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Shuffle to interleave fraudulent and legitimate transactions
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[SYNTHETIC-DATA] Generated {len(df)} synthetic rows ({n_fraud} fraud, {n_legit} legitimate). Saved to: {save_path}")

    return df


if __name__ == "__main__":
    df_synthetic = generate_synthetic_fraud_data()
    print("Columns:", list(df_synthetic.columns))
    print("Class distribution:\n", df_synthetic[TARGET_COLUMN].value_counts())
