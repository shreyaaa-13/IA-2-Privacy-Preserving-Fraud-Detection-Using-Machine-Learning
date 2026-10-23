r"""
Experiment F (Module 5): Adversarial Compressive Privacy Autoencoder.
Implements a deep representation learner that formalizes the Information Bottleneck principle:
$$\min_{\theta_E, \theta_T} \max_{\theta_A} \left[ \mathcal{L}_{task}(Y, T(E(X))) - \lambda \mathcal{L}_{adv}(S, A(E(X))) \right]$$

Conceptual Framing:
- Task Utility $Y$: Binary fraud classification label.
- Sensitive Proxy Attribute $S$: Binarized transaction amount tier (e.g. High-net-worth tier > 85th percentile).
  Financial relevance: Cardholder wealth profile and account tier must remain confidential while detecting fraud.
- Latent Representation $Z = E(X)$: Compressed feature vector of dimension $d \in \{4, 8, 16\}$.
- Adversary $A(Z)$: Strong neural adversary attempting to reconstruct sensitive attribute $S$ from representation $Z$.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from configs.config import COMPRESSIVE_PRIVACY_CONFIG, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED
from src.evaluation import compute_classification_metrics, ResourceTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural Architectures for Compressive Privacy
# ---------------------------------------------------------------------------

class CompressiveEncoder(nn.Module):
    """Compresses raw 29-dimensional transaction vector into latent representation Z."""

    def __init__(self, input_dim: int = 29, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskPredictor(nn.Module):
    """Predicts fraud utility label Y from compressed representation Z."""

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SensitiveAdversary(nn.Module):
    """Adversary attempting to infer sensitive attribute S from compressed representation Z."""

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def train_adversarial_compressive_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    latent_dim: int = 8,
    lambda_adv: float = 0.5,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 0.005,
    random_state: int = RANDOM_SEED
) -> Dict[str, Any]:
    r"""
    Train Adversarial Compressive Representation via Alternating Min-Max Optimization.

    Alternating Protocol:
    1. Update Adversary A to minimize binary cross-entropy on sensitive attribute S:
       \mathcal{L}_{adv} = BCEWithLogits(A(Z.detach()), S)
    2. Update Encoder E and Task Predictor T to minimize:
       \mathcal{L}_{task} - \lambda \mathcal{L}_{adv}(A(Z), S)
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    input_dim = X_train.shape[1]

    # Convert to Tensors
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    s_tr_t = torch.tensor(s_train, dtype=torch.float32).unsqueeze(1)

    X_te_t = torch.tensor(X_test, dtype=torch.float32)

    # Class weighting for task loss
    pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())], dtype=torch.float32)
    criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_adv = nn.BCEWithLogitsLoss()

    encoder = CompressiveEncoder(input_dim=input_dim, latent_dim=latent_dim)
    task_pred = TaskPredictor(latent_dim=latent_dim)
    adversary = SensitiveAdversary(latent_dim=latent_dim)

    opt_task = torch.optim.Adam(list(encoder.parameters()) + list(task_pred.parameters()), lr=lr)
    opt_adv = torch.optim.Adam(adversary.parameters(), lr=lr)

    dataset = TensorDataset(X_tr_t, y_tr_t, s_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    with ResourceTracker() as tracker:
        for epoch in range(epochs):
            for bx, by, bs in loader:
                # Step 1: Train Adversary
                opt_adv.zero_grad()
                with torch.no_grad():
                    z_detached = encoder(bx)
                pred_s_adv = adversary(z_detached)
                loss_adv = criterion_adv(pred_s_adv, bs)
                loss_adv.backward()
                opt_adv.step()

                # Step 2: Train Encoder + Task Predictor (Min-Max)
                opt_task.zero_grad()
                z = encoder(bx)
                pred_y = task_pred(z)
                pred_s_from_z = adversary(z)

                loss_t = criterion_task(pred_y, by)
                loss_a = criterion_adv(pred_s_from_z, bs)

                # Penalize adversary ability to predict S
                total_task_loss = loss_t - lambda_adv * loss_a
                total_task_loss.backward()
                opt_task.step()

    # Evaluation on Test Data
    encoder.eval()
    task_pred.eval()
    adversary.eval()

    with torch.no_grad():
        z_test = encoder(X_te_t)
        logits_y = task_pred(z_test).squeeze()
        logits_s = adversary(z_test).squeeze()

        probs_y = torch.sigmoid(logits_y).numpy()
        probs_s = torch.sigmoid(logits_s).numpy()

        preds_y = (probs_y >= 0.50).astype(int)

    m = compute_classification_metrics(y_test, preds_y, probs_y, f"Adversarial_Compressive_d{latent_dim}_lam{lambda_adv}")
    m.update(tracker.get_summary())

    adv_auc = float(roc_auc_score(s_test, probs_s)) if len(np.unique(s_test)) > 1 else 0.50
    m["latent_dimension"] = latent_dim
    m["lambda_adversary"] = lambda_adv
    m["adversary_sensitive_auc"] = round(adv_auc, 4)
    m["privacy_or_attack_metric"] = f"Dim={latent_dim}, $\\lambda={lambda_adv}$ | Adv AUC(S): {adv_auc:.4f}"
    m["notes"] = (
        f"Adversarial autoencoder with latent dimension {latent_dim} and penalty lambda={lambda_adv}. "
        f"Adversary AUC measures leakage of sensitive amount tier proxy S."
    )

    return m


def run_compressive_privacy_demonstration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    latent_dimensions: Optional[List[int]] = None,
    save_outputs: bool = True
) -> Dict[str, Any]:
    """
    Execute full empirical Compressive Privacy experiment:
    1. Define sensitive proxy attribute S (High-value transaction > 85th percentile).
    2. Remove Amount directly from X to measure non-linear latent leakage.
    3. Evaluate Adversarial Autoencoder across latent dimensions d in [4, 8, 16] and lambda in [0.0, 0.5].
    """
    if latent_dimensions is None:
        latent_dimensions = COMPRESSIVE_PRIVACY_CONFIG["latent_dimensions"]

    amount_idx = feature_names.index("Amount") if "Amount" in feature_names else 0
    s_thresh = np.percentile(X_train[:, amount_idx], 85)
    s_train = (X_train[:, amount_idx] > s_thresh).astype(int)
    s_test = (X_test[:, amount_idx] > s_thresh).astype(int)

    # Remove Amount from inputs to evaluate latent proxy recovery
    non_s_indices = [i for i in range(X_train.shape[1]) if i != amount_idx]
    X_tr_no_s = X_train[:, non_s_indices]
    X_te_no_s = X_test[:, non_s_indices]

    results_table = []

    # Sweep across latent dimensions with and without adversarial penalty
    for d in latent_dimensions:
        # Non-adversarial baseline (lambda = 0.0)
        res_baseline = train_adversarial_compressive_model(
            X_tr_no_s, y_train, s_train, X_te_no_s, y_test, s_test,
            latent_dim=d, lambda_adv=0.0, epochs=5
        )
        results_table.append(res_baseline)

        # Adversarial compressive model (lambda = 0.5)
        res_adv = train_adversarial_compressive_model(
            X_tr_no_s, y_train, s_train, X_te_no_s, y_test, s_test,
            latent_dim=d, lambda_adv=COMPRESSIVE_PRIVACY_CONFIG["lambda_adversary"], epochs=5
        )
        results_table.append(res_adv)

    results_df = pd.DataFrame(results_table)

    summary = {
        "module": "Module 5: Compressive Privacy for Machine Learning",
        "sensitive_proxy": "Transaction Amount Tier (> 85th percentile)",
        "objective": "min [L_task(Y, T(Z)) - lambda * L_adv(S, A(Z))]",
        "experiments": results_table
    }

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(RESULTS_DIR / "compressive_privacy_comparison.csv", index=False)
        with open(RESULTS_DIR / "compressive_privacy_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

    return summary


if __name__ == "__main__":
    from src.data_preprocessing import load_dataset, split_data, preprocess_features
    df, _ = load_dataset()
    train_df, test_df, val_df, _ = split_data(df)
    X_tr, X_te, _, y_tr, y_te, _, feats, _ = preprocess_features(train_df, test_df, val_df)
    comp_out = run_compressive_privacy_demonstration(X_tr, y_tr, X_te, y_te, feature_names=feats, latent_dimensions=[4, 8])
    print(pd.DataFrame(comp_out["experiments"])[["experiment_name", "latent_dimension", "lambda_adversary", "pr_auc", "adversary_sensitive_auc"]])
