"""
End-to-End Pipeline Integration Tests for IA 2.
Verifies baseline with threshold tuning, comparative MIA, DP-SGD with Opacus,
LDP, PPDM microaggregation, and Adversarial Compressive Privacy.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_data import generate_synthetic_fraud_data
from src.data_preprocessing import split_data, preprocess_features
from src.baseline_model import BaselineFraudDetector, run_baseline_experiment
from src.evaluation import find_optimal_threshold
from src.membership_inference import run_comparative_mia_experiment
from src.differential_privacy import run_dp_privacy_utility_sweep, train_dp_sgd_opacus
from src.local_differential_privacy import run_ldp_demonstration
from src.privacy_preserving_mining import run_ppdm_demonstration
from src.compressive_privacy import run_compressive_privacy_demonstration


@pytest.fixture(scope="module")
def prepared_data():
    """Generate small fixture dataset and preprocessed arrays."""
    df = generate_synthetic_fraud_data(n_samples=600, fraud_ratio=0.05, random_state=42, save_path=None)
    train_df, test_df, val_df, _ = split_data(df, method="stratified", test_size=0.2, val_size=0.1)
    X_tr, X_te, X_v, y_tr, y_te, y_v, feats, scaler = preprocess_features(train_df, test_df, val_df)
    return {
        "X_tr": X_tr, "X_te": X_te, "X_v": X_v,
        "y_tr": y_tr, "y_te": y_te, "y_v": y_v,
        "features": feats, "df": df
    }


def test_baseline_and_threshold_tuning(prepared_data):
    """Test baseline model fit, predict, and validation-only threshold tuning."""
    X_tr, y_tr = prepared_data["X_tr"], prepared_data["y_tr"]
    X_v, y_v = prepared_data["X_v"], prepared_data["y_v"]
    X_te, y_te = prepared_data["X_te"], prepared_data["y_te"]

    model = BaselineFraudDetector({"n_estimators": 10, "max_depth": 5, "random_state": 42, "n_jobs": 1})
    model.fit(X_tr, y_tr)

    p_val = model.predict_proba(X_v)[:, 1]
    best_t, best_f1, meta = find_optimal_threshold(y_v, p_val, metric="f1")

    assert 0.0 < best_t < 1.0
    assert "protocol" in meta

    preds_tuned = model.predict(X_te, threshold=best_t)
    assert len(preds_tuned) == len(X_te)


def test_comparative_membership_inference(prepared_data):
    """Test comparative MIA against baseline and DP models."""
    X_tr, y_tr = prepared_data["X_tr"], prepared_data["y_tr"]
    mia_out = run_comparative_mia_experiment(X_tr, y_tr, save_outputs=False)

    assert "comparison_table" in mia_out
    assert len(mia_out["comparison_table"]) >= 3
    for auc in mia_out["comparison_table"]["attack_auc"]:
        assert 0.0 <= auc <= 1.0


def test_dp_diffprivlib_and_dpsgd(prepared_data):
    """Test IBM diffprivlib sweep and PyTorch/Opacus DP-SGD."""
    X_tr, y_tr = prepared_data["X_tr"], prepared_data["y_tr"]
    X_te, y_te = prepared_data["X_te"], prepared_data["y_te"]

    # diffprivlib sweep
    sweep_df, _ = run_dp_privacy_utility_sweep(
        X_tr, y_tr, X_te, y_te, epsilons=[1.0, 5.0], data_norm=5.0, save_outputs=False
    )
    assert len(sweep_df) == 3  # Ref + 2 epsilons

    # Opacus DP-SGD
    dp_m, ref_m = train_dp_sgd_opacus(
        X_tr, y_tr, X_te, y_te, target_epsilon=10.0, epochs=2, batch_size=128, save_outputs=False
    )
    assert "achieved_epsilon" in dp_m
    assert dp_m["achieved_epsilon"] > 0.0


def test_ldp_pipeline(prepared_data):
    """Test Local Differential Privacy demonstration."""
    X_tr, y_tr = prepared_data["X_tr"], prepared_data["y_tr"]
    X_te, y_te = prepared_data["X_te"], prepared_data["y_te"]

    res = run_ldp_demonstration(X_tr, y_tr, X_te, y_te, feature_index=0, epsilons=[1.0], save_outputs=False)
    assert len(res["experiments"]) == 1
    assert "wasserstein_distance" in res["experiments"][0]


def test_ppdm_pipeline(prepared_data):
    """Test PPDM microaggregation."""
    X_tr, y_tr = prepared_data["X_tr"], prepared_data["y_tr"]
    X_te, y_te = prepared_data["X_te"], prepared_data["y_te"]

    res = run_ppdm_demonstration(X_tr, y_tr, X_te, y_te, k_values=[5], save_outputs=False)
    assert "pca_vs_dp_clarification" in res
    assert len(res["experiments"]) == 2  # Baseline + k=5


def test_adversarial_compressive_privacy(prepared_data):
    """Test Adversarial Compressive Privacy Autoencoder."""
    X_tr, y_tr = prepared_data["X_tr"], prepared_data["y_tr"]
    X_te, y_te = prepared_data["X_te"], prepared_data["y_te"]
    feats = prepared_data["features"]

    res = run_compressive_privacy_demonstration(
        X_tr, y_tr, X_te, y_te, feature_names=feats, latent_dimensions=[4], save_outputs=False
    )
    assert len(res["experiments"]) == 2  # lambda=0.0 and lambda=0.5
    assert "adversary_sensitive_auc" in res["experiments"][0]
