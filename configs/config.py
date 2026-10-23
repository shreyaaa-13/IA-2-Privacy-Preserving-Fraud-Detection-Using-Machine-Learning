"""
Centralized Configuration Module for IA 2: Privacy-Preserving Fraud Detection.
Supports multi-seed runs, validation threshold tuning, DP-SGD with Opacus,
Adversarial Compressive Privacy Autoencoder, and resource monitoring.
"""

import os
from pathlib import Path
import numpy as np

# Base Paths (relative to this file)
CONFIGS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIGS_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Dataset Files
REAL_DATASET_NAME = "creditcard.csv"
REAL_DATASET_PATH = DATA_DIR / REAL_DATASET_NAME

SYNTHETIC_DATASET_NAME = "synthetic_creditcard.csv"
SYNTHETIC_DATASET_PATH = DATA_DIR / SYNTHETIC_DATASET_NAME

# Provenance Constants
DATASET_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
DATASET_EXPECTED_ROWS = 284807
DATASET_EXPECTED_COLS = 31
DATASET_FRAUD_COUNT = 492
DATASET_LEGIT_COUNT = 284315

# Global Reproducibility Seeds
RANDOM_SEED = 42
MULTI_SEEDS = [42, 123, 456]

# Preprocessing & Data Partitioning
TEST_SPLIT_RATIO = 0.20
VAL_SPLIT_RATIO = 0.10
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
TARGET_COLUMN = "Class"

# Baseline Random Forest Configuration
BASELINE_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# Threshold Tuning Configuration (Validation-only grid search)
THRESHOLD_SEARCH_GRID = np.linspace(0.01, 0.99, 99)

# Membership Inference Attack (MIA) Configuration
MIA_PARAMS = {
    "shadow_dataset_ratio": 0.50,
    "random_state": RANDOM_SEED,
    "attack_model": "logistic_regression",
    "threshold_metric": "confidence"
}

# Central Differential Privacy (DP Logistic Regression) Configuration
DP_CONFIG = {
    "epsilons": [0.1, 0.5, 1.0, 5.0, 10.0],
    "default_epsilon": 1.0,
    "data_norm": 5.0,  # L2 clipping bound C for DP logistic regression
    "random_state": RANDOM_SEED
}

# PyTorch & Opacus DP-SGD Configuration
DP_SGD_CONFIG = {
    "batch_size": 512,
    "max_grad_norm": 1.0,  # Per-sample gradient clipping threshold C
    "noise_multiplier": 1.0,  # Calibrated Gaussian noise scale
    "target_epsilon": 5.0,
    "target_delta": 1e-5,  # Strictly < 1/N where N ≈ 200,000
    "epochs": 5,
    "learning_rate": 0.01,
    "random_state": RANDOM_SEED
}

# Local Differential Privacy (LDP) Configuration
LDP_CONFIG = {
    "target_feature": "Amount",
    "epsilons": [0.5, 1.0, 2.0, 5.0],
    "default_epsilon": 1.0,
    "random_state": RANDOM_SEED
}

# PPDM Microaggregation Configuration
PPDM_CONFIG = {
    "k_values": [5, 10],
    "random_state": RANDOM_SEED
}

# Adversarial Compressive Privacy Autoencoder Configuration
COMPRESSIVE_PRIVACY_CONFIG = {
    "latent_dimensions": [4, 8, 16],
    "default_dimension": 8,
    "lambda_adversary": 0.5,  # Weight on adversary penalty in min-max objective
    "epochs": 10,
    "batch_size": 512,
    "learning_rate": 0.005,
    "random_state": RANDOM_SEED
}

def ensure_directories():
    """Ensure all required output directories exist."""
    for directory in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Real Dataset Expected Path: {REAL_DATASET_PATH}")
    print(f"Dataset exists on disk: {REAL_DATASET_PATH.exists()}")
