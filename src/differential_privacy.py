r"""
Experiment C (Module 2): Differential Privacy and DP-SGD Implementation.
Provides two mathematically grounded differential privacy implementations:
1. Pure $\epsilon$-DP Logistic Regression via IBM diffprivlib (objective perturbation with $L_2$ clipping bound $C=5.0$).
2. DP-SGD using PyTorch and Opacus (per-sample gradient clipping, Gaussian noise addition, and Rényi Differential Privacy accounting).
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import diffprivlib
from diffprivlib.models import LogisticRegression as DPLogisticRegression
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as StandardLogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

from configs.config import DP_CONFIG, DP_SGD_CONFIG, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED
from src.evaluation import compute_classification_metrics, ResourceTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. IBM diffprivlib Pure Differential Privacy Sweep
# ---------------------------------------------------------------------------

def run_dp_privacy_utility_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilons: Optional[List[float]] = None,
    data_norm: float = 5.0,
    save_outputs: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""Empirically evaluate the Privacy-Utility Trade-Off across varying $\epsilon$ budgets using diffprivlib."""
    if epsilons is None:
        epsilons = DP_CONFIG["epsilons"]

    # Reference Non-Private Logistic Regression
    with ResourceTracker() as ref_tracker:
        non_dp_model = StandardLogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
        non_dp_model.fit(X_train, y_train)
        y_pred_ref = non_dp_model.predict(X_test)
        y_prob_ref = non_dp_model.predict_proba(X_test)[:, 1]

    ref_metrics = compute_classification_metrics(y_test, y_pred_ref, y_prob_ref, "Standard_LogisticRegression_NonPrivate")
    ref_metrics.update(ref_tracker.get_summary())
    ref_metrics["privacy_or_attack_metric"] = "None (Non-Private Reference)"
    ref_metrics["notes"] = "Standard Logistic Regression without DP noise."

    sweep_records = [ref_metrics]

    for eps in epsilons:
        with ResourceTracker() as tracker:
            dp_clf = DPLogisticRegression(
                epsilon=eps,
                data_norm=data_norm,
                random_state=RANDOM_SEED,
                max_iter=1000
            )
            dp_clf.fit(X_train, y_train)
            y_pred_dp = dp_clf.predict(X_test)
            y_prob_dp = dp_clf.predict_proba(X_test)[:, 1]

        m = compute_classification_metrics(y_test, y_pred_dp, y_prob_dp, f"DP_LogisticRegression_eps_{eps}")
        m.update(tracker.get_summary())
        m["epsilon"] = eps
        m["data_norm_clipping"] = data_norm
        m["diffprivlib_version"] = diffprivlib.__version__
        m["privacy_or_attack_metric"] = f"Pure $\\epsilon$-DP ($\\epsilon={eps}, C={data_norm}$)"
        m["notes"] = f"diffprivlib objective perturbation with L2 norm bound C={data_norm}."
        sweep_records.append(m)

    sweep_df = pd.DataFrame(sweep_records)

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(RESULTS_DIR / "dp_epsilon_sweep_comparison.csv", index=False)
        with open(RESULTS_DIR / "differential_privacy_results.json", "w", encoding="utf-8") as f:
            json.dump({"ref": ref_metrics, "sweep": sweep_records}, f, indent=4)

    return sweep_df, {"ref": ref_metrics, "dp_sweep_results": sweep_records}


# ---------------------------------------------------------------------------
# 2. PyTorch & Opacus DP-SGD Neural Network Implementation
# ---------------------------------------------------------------------------

class FraudDetectionMLP(nn.Module):
    """Feedforward Neural Network for binary fraud classification."""

    def __init__(self, input_dim: int = 29, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_dp_sgd_opacus(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_epsilon: float = 5.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 0.01,
    random_state: int = RANDOM_SEED,
    save_outputs: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    r"""
    Train a binary fraud classification neural network with DP-SGD using PyTorch and Opacus.

    Core Mathematical Operations:
    1. Per-Sample Gradient Clipping: For each sample i in batch B, clip gradient:
       \bar{g}_i = g_i / \max(1, \|g_i\|_2 / C)
    2. Calibrated Noise Addition:
       \tilde{g} = \frac{1}{|B|} \left( \sum_{i \in B} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) \right)
    3. Rényi Differential Privacy (RDP) Accounting:
       Composes privacy loss across iterations and converts to (\epsilon, \delta)-DP.

    Returns:
    --------
    dp_sgd_metrics : Dict[str, Any]
    non_private_nn_metrics : Dict[str, Any]
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    input_dim = X_train.shape[1]

    # Convert to PyTorch Tensors
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)

    # Class weight to counteract severe class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train Non-Private PyTorch Reference Model
    logger.info("Training Non-Private Reference PyTorch Neural Network...")
    ref_model = FraudDetectionMLP(input_dim=input_dim)
    ref_opt = torch.optim.Adam(ref_model.parameters(), lr=lr)
    ref_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    with ResourceTracker() as ref_tracker:
        ref_model.train()
        for epoch in range(epochs):
            for bx, by in ref_loader:
                ref_opt.zero_grad()
                logits = ref_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                ref_opt.step()

    ref_model.eval()
    with torch.no_grad():
        test_logits_ref = ref_model(X_te_t).squeeze()
        test_probs_ref = torch.sigmoid(test_logits_ref).numpy()
        test_preds_ref = (test_probs_ref >= 0.50).astype(int)

    ref_nn_metrics = compute_classification_metrics(y_test, test_preds_ref, test_probs_ref, "PyTorch_MLP_NonPrivate")
    ref_nn_metrics.update(ref_tracker.get_summary())
    ref_nn_metrics["privacy_or_attack_metric"] = "None (Non-Private PyTorch Baseline)"
    ref_nn_metrics["notes"] = "Reference neural network trained without gradient noise."

    # Train Differentially Private Model with Opacus
    logger.info(f"Training DP-SGD Neural Network with Opacus (Target epsilon={target_epsilon}, delta={target_delta})...")
    dp_model = FraudDetectionMLP(input_dim=input_dim)
    dp_opt = torch.optim.Adam(dp_model.parameters(), lr=lr)
    dp_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    privacy_engine = PrivacyEngine()
    dp_model, dp_opt, dp_loader = privacy_engine.make_private_with_epsilon(
        module=dp_model,
        optimizer=dp_opt,
        data_loader=dp_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm
    )

    with ResourceTracker() as dp_tracker:
        dp_model.train()
        for epoch in range(epochs):
            for bx, by in dp_loader:
                dp_opt.zero_grad()
                logits = dp_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                dp_opt.step()

    # Query achieved epsilon from RDP accountant
    achieved_eps = float(privacy_engine.get_epsilon(delta=target_delta))
    logger.info(f"DP-SGD Complete: Achieved (epsilon={achieved_eps:.4f}, delta={target_delta})")

    dp_model.eval()
    with torch.no_grad():
        test_logits_dp = dp_model(X_te_t).squeeze()
        test_probs_dp = torch.sigmoid(test_logits_dp).numpy()
        test_preds_dp = (test_probs_dp >= 0.50).astype(int)

    dp_sgd_metrics = compute_classification_metrics(y_test, test_preds_dp, test_probs_dp, "PyTorch_Opacus_DPSGD")
    dp_sgd_metrics.update(dp_tracker.get_summary())
    dp_sgd_metrics["achieved_epsilon"] = round(achieved_eps, 4)
    dp_sgd_metrics["target_delta"] = target_delta
    dp_sgd_metrics["max_grad_norm"] = max_grad_norm
    dp_sgd_metrics["privacy_or_attack_metric"] = f"DP-SGD ($\\epsilon={achieved_eps:.2f}, \\delta={target_delta}$)"
    dp_sgd_metrics["notes"] = f"PyTorch/Opacus DP-SGD with per-sample clipping C={max_grad_norm} and RDP accounting."

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "dp_sgd_opacus_results.json", "w", encoding="utf-8") as f:
            json.dump({"dp_sgd": dp_sgd_metrics, "reference_nn": ref_nn_metrics}, f, indent=4)

    return dp_sgd_metrics, ref_nn_metrics


if __name__ == "__main__":
    from src.data_preprocessing import load_dataset, split_data, preprocess_features
    df, _ = load_dataset()
    train_df, test_df, val_df, _ = split_data(df)
    X_tr, X_te, _, y_tr, y_te, _, _, _ = preprocess_features(train_df, test_df, val_df)
    sweep_df, _ = run_dp_privacy_utility_sweep(X_tr, y_tr, X_te, y_te, epsilons=[1.0, 5.0])
    dp_m, ref_m = train_dp_sgd_opacus(X_tr, y_tr, X_te, y_te, epochs=2)
    print("DP-SGD Metrics:", dp_m)
