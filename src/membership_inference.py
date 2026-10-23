r"""
Experiment B (Module 1): Controlled Membership Inference Attack (MIA).
Evaluates privacy leakage against both non-private baseline and differentially private models
across varying privacy budgets $\epsilon \in \{0.1, 1.0, 10.0\}$.

Conceptual Formulation:
- Target Classification Task: $f(x) \rightarrow y \in \{0, 1\}$
- Attacker's Membership Task: $\mathcal{A}(x, y, f) \rightarrow m \in \{0, 1\}$
  where $m=1$ if $(x, y) \in D_{target\_train}$ (Member), $m=0$ if $(x, y) \notin D_{target\_train}$ (Non-Member).

Key IA 2 Extension:
Evaluates attack success against both non-private target models and DP target models
to demonstrate empirically that tightening the privacy budget $\epsilon$ suppresses attack advantage.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
from diffprivlib.models import LogisticRegression as DPLogisticRegression

from configs.config import RESULTS_DIR, FIGURES_DIR, RANDOM_SEED
from src.evaluation import ResourceTracker, plot_mia_comparison_curve

logger = logging.getLogger(__name__)


def extract_attack_features(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    clip_eps: float = 1e-7
) -> np.ndarray:
    """Extract prediction confidence, entropy, cross-entropy loss, and margin."""
    probs = model.predict_proba(X)
    probs = np.clip(probs, clip_eps, 1.0 - clip_eps)

    n_samples = len(X)
    true_class_probs = probs[np.arange(n_samples), y]
    max_probs = np.max(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    loss = -np.log(true_class_probs)

    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    return np.column_stack([true_class_probs, max_probs, entropy, loss, margin])


def partition_membership_datasets(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_SEED
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Strictly partition into 4 disjoint sets: Target Train, Target Holdout, Shadow Train, Shadow Holdout."""
    n_samples = len(X)
    indices = np.arange(n_samples)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)

    q1 = int(n_samples * 0.25)
    q2 = int(n_samples * 0.50)
    q3 = int(n_samples * 0.75)

    return {
        "target_train": (X[indices[:q1]], y[indices[:q1]]),
        "target_holdout": (X[indices[q1:q2]], y[indices[q1:q2]]),
        "shadow_train": (X[indices[q2:q3]], y[indices[q2:q3]]),
        "shadow_holdout": (X[indices[q3:]], y[indices[q3:]]),
    }


def evaluate_mia_on_target_model(
    target_model: Any,
    shadow_model: Any,
    partitions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_name: str = "Target_Model"
) -> Dict[str, Any]:
    """Train shadow attack classifier and evaluate against target model."""
    X_tgt_tr, y_tgt_tr = partitions["target_train"]
    X_tgt_ho, y_tgt_ho = partitions["target_holdout"]
    X_shd_tr, y_shd_tr = partitions["shadow_train"]
    X_shd_ho, y_shd_ho = partitions["shadow_holdout"]

    # Extract shadow features for attack training
    feat_shd_in = extract_attack_features(shadow_model, X_shd_tr, y_shd_tr)
    feat_shd_out = extract_attack_features(shadow_model, X_shd_ho, y_shd_ho)

    X_atk_tr = np.vstack([feat_shd_in, feat_shd_out])
    y_atk_tr = np.concatenate([np.ones(len(feat_shd_in), dtype=int), np.zeros(len(feat_shd_out), dtype=int)])

    # Fit Attack Classifier
    attack_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    attack_clf.fit(X_atk_tr, y_atk_tr)

    # Extract target features for attack evaluation (50% members, 50% non-members)
    feat_tgt_in = extract_attack_features(target_model, X_tgt_tr, y_tgt_tr)
    feat_tgt_out = extract_attack_features(target_model, X_tgt_ho, y_tgt_ho)

    X_atk_te = np.vstack([feat_tgt_in, feat_tgt_out])
    y_atk_te = np.concatenate([np.ones(len(feat_tgt_in), dtype=int), np.zeros(len(feat_tgt_out), dtype=int)])

    attack_preds = attack_clf.predict(X_atk_te)
    attack_probs = attack_clf.predict_proba(X_atk_te)[:, 1]

    attack_auc = float(roc_auc_score(y_atk_te, attack_probs))
    attack_acc = float(accuracy_score(y_atk_te, attack_preds))
    attack_prec = float(precision_score(y_atk_te, attack_preds, zero_division=0))
    attack_rec = float(recall_score(y_atk_te, attack_preds, zero_division=0))
    attack_f1 = float(f1_score(y_atk_te, attack_preds, zero_division=0))

    fpr, tpr, _ = roc_curve(y_atk_te, attack_probs)
    advantage = float(2 * (attack_acc - 0.50)) if attack_acc >= 0.50 else 0.0

    return {
        "model_name": model_name,
        "attack_auc": round(attack_auc, 4),
        "attack_accuracy": round(attack_acc, 4),
        "attacker_advantage": round(advantage, 4),
        "attack_precision": round(attack_prec, 4),
        "attack_recall": round(attack_rec, 4),
        "attack_f1": round(attack_f1, 4),
        "fpr": fpr,
        "tpr": tpr
    }


def run_comparative_mia_experiment(
    X: np.ndarray,
    y: np.ndarray,
    save_outputs: bool = True
) -> Dict[str, Any]:
    """
    Execute controlled Membership Inference Attack comparing:
    1. Baseline Non-Private Random Forest
    2. Non-Private Logistic Regression Reference
    3. DP-Logistic Regression at epsilon = 10.0 (weak DP)
    4. DP-Logistic Regression at epsilon = 1.0 (standard DP)
    5. DP-Logistic Regression at epsilon = 0.1 (strong DP)
    """
    partitions = partition_membership_datasets(X, y)
    X_tgt_tr, y_tgt_tr = partitions["target_train"]
    X_shd_tr, y_shd_tr = partitions["shadow_train"]

    comparison_results = []

    # 1. Non-Private Random Forest
    rf_tgt = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1).fit(X_tgt_tr, y_tgt_tr)
    rf_shd = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_SEED + 1, n_jobs=-1).fit(X_shd_tr, y_shd_tr)
    res_rf = evaluate_mia_on_target_model(rf_tgt, rf_shd, partitions, "Baseline Random Forest (Non-Private)")
    comparison_results.append(res_rf)

    # 2. Non-Private Logistic Regression Reference
    lr_tgt = LogisticRegression(max_iter=500, random_state=RANDOM_SEED).fit(X_tgt_tr, y_tgt_tr)
    lr_shd = LogisticRegression(max_iter=500, random_state=RANDOM_SEED + 1).fit(X_shd_tr, y_shd_tr)
    res_lr = evaluate_mia_on_target_model(lr_tgt, lr_shd, partitions, "Logistic Regression (Non-Private Ref)")
    comparison_results.append(res_lr)

    # 3. Differentially Private Logistic Regression Models
    for eps in [10.0, 1.0, 0.1]:
        dp_tgt = DPLogisticRegression(epsilon=eps, data_norm=5.0, random_state=RANDOM_SEED, max_iter=500).fit(X_tgt_tr, y_tgt_tr)
        dp_shd = DPLogisticRegression(epsilon=eps, data_norm=5.0, random_state=RANDOM_SEED + 1, max_iter=500).fit(X_shd_tr, y_shd_tr)
        res_dp = evaluate_mia_on_target_model(dp_tgt, dp_shd, partitions, f"DP-LogisticReg (eps={eps})")
        comparison_results.append(res_dp)

    summary_records = []
    for r in comparison_results:
        summary_records.append({
            "experiment_name": f"MIA_vs_{r['model_name']}",
            "model_evaluated": r["model_name"],
            "attack_auc": r["attack_auc"],
            "attack_accuracy": r["attack_accuracy"],
            "random_baseline_acc": 0.5000,
            "attacker_advantage": r["attacker_advantage"],
            "attack_precision": r["attack_precision"],
            "attack_recall": r["attack_recall"],
            "attack_f1_score": r["attack_f1"],
            "privacy_or_attack_metric": f"MIA AUC: {r['attack_auc']:.4f} | Adv: {r['attacker_advantage']:.4f}",
            "notes": "4-way disjoint split with 50/50 balanced member/non-member evaluation."
        })

    summary_df = pd.DataFrame(summary_records)

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(RESULTS_DIR / "mia_comparative_results.csv", index=False)
        with open(RESULTS_DIR / "membership_inference_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary_records, f, indent=4)

        plot_mia_comparison_curve(
            comparison_results,
            save_path=FIGURES_DIR / "mia_comparison_across_models.png"
        )
        logger.info("MIA comparative evaluation complete and artifacts saved.")

    return {
        "comparison_table": summary_df,
        "raw_results": comparison_results,
        "primary_baseline_result": summary_records[0]
    }


if __name__ == "__main__":
    from src.data_preprocessing import load_dataset, split_data, preprocess_features
    df, _ = load_dataset()
    train_df, test_df, val_df, _ = split_data(df)
    X_tr, _, _, y_tr, _, _, _, _ = preprocess_features(train_df, test_df, val_df)
    mia_out = run_comparative_mia_experiment(X_tr, y_tr)
    print("\n--- MIA Comparative Summary ---")
    print(mia_out["comparison_table"][["model_evaluated", "attack_auc", "attack_accuracy", "attacker_advantage"]].to_string(index=False))
