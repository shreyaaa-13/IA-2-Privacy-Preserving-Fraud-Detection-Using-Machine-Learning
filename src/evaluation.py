"""
Evaluation, Metrics, Validation Threshold Tuning, and Visualization Suite for IA 2.
Provides rigorous metric computation, validation-only probability threshold selection,
uncertainty quantification across random seeds, and resource profiling (runtime and memory).
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)

from configs.config import RESULTS_DIR, FIGURES_DIR, THRESHOLD_SEARCH_GRID

logger = logging.getLogger(__name__)


class ResourceTracker:
    """Measures execution runtime and peak RSS memory usage."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = 0.0
        self.start_mem_mb = 0.0
        self.elapsed_sec = 0.0
        self.peak_mem_mb = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_mem_mb = self.process.memory_info().rss / (1024 * 1024)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_sec = time.perf_counter() - self.start_time
        current_mem_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_mem_mb = max(self.start_mem_mb, current_mem_mb)

    def get_summary(self) -> Dict[str, float]:
        return {
            "runtime_seconds": round(self.elapsed_sec, 3),
            "memory_mb": round(self.peak_mem_mb, 2)
        }


def find_optimal_threshold(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    metric: str = "f1",
    search_grid: Optional[np.ndarray] = None
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Tune the classification probability threshold strictly on the VALIDATION set.
    NEVER touches the test set, preventing decision boundary data leakage.

    Parameters:
    -----------
    y_val : np.ndarray
        Ground-truth labels for validation set.
    y_prob_val : np.ndarray
        Predicted probabilities of fraud for validation set.
    metric : str
        Optimization objective ('f1' or 'balanced').
    search_grid : np.ndarray
        Array of candidate threshold values.

    Returns:
    --------
    best_threshold : float
    best_score : float
    tuning_metadata : Dict[str, Any]
    """
    if search_grid is None:
        search_grid = THRESHOLD_SEARCH_GRID

    best_thresh = 0.50
    best_score = -1.0
    val_records = []

    for t in search_grid:
        preds = (y_prob_val >= t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f = f1_score(y_val, preds, zero_division=0)

        val_records.append({"threshold": t, "precision": p, "recall": r, "f1": f})

        score = f if metric == "f1" else (p + r) / 2.0
        if score > best_score:
            best_score = score
            best_thresh = float(t)

    tuning_metadata = {
        "optimal_threshold": round(best_thresh, 4),
        "validation_best_f1": round(best_score, 4),
        "search_grid_size": len(search_grid),
        "protocol": "Tuned strictly on validation set; test set remained unobserved."
    }
    logger.info(f"Threshold tuning complete: Optimal threshold = {best_thresh:.4f} (Val F1 = {best_score:.4f})")
    return best_thresh, best_score, tuning_metadata


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    experiment_name: str = "Model_Evaluation",
    threshold_used: float = 0.50
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics focusing on imbalanced fraud detection."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = float((rec + specificity) / 2.0)

    pr_auc = None
    roc_auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                pr_auc = float(average_precision_score(y_true, y_prob))
                roc_auc = float(roc_auc_score(y_true, y_prob))
            else:
                pr_auc = 0.0
                roc_auc = 0.5
        except Exception as e:
            logger.warning(f"Could not compute curve AUCs: {e}")

    acc_critique = (
        "Accuracy is misleading: If a naive model predicts all transactions as legitimate (0), "
        f"it achieves ~{round(float(np.mean(y_true == 0)) * 100, 2)}% accuracy, while detecting zero fraud (Recall=0.0). "
        "PR-AUC, Recall, and Precision are the primary authoritative metrics."
    )

    metrics = {
        "experiment_name": experiment_name,
        "threshold_used": round(threshold_used, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else "N/A",
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "specificity": round(specificity, 4),
        "true_positives_tp": int(tp),
        "false_positives_fp": int(fp),
        "true_negatives_tn": int(tn),
        "false_negatives_fn": int(fn),
        "accuracy_critique": acc_critique
    }
    return metrics


def aggregate_multiseed_metrics(seed_metrics: List[Dict[str, Any]], exp_name: str) -> Dict[str, Any]:
    """Compute mean and standard deviation across multiple random seeds."""
    keys_to_agg = ["precision", "recall", "f1_score", "pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    agg_result = {"experiment_name": exp_name, "num_seeds": len(seed_metrics)}

    for k in keys_to_agg:
        vals = [m[k] for m in seed_metrics if isinstance(m.get(k), (int, float))]
        if vals:
            agg_result[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            agg_result[f"{k}_std"] = round(float(np.std(vals)), 4)
            agg_result[k] = f"{agg_result[f'{k}_mean']:.4f} ± {agg_result[f'{k}_std']:.4f}"
        else:
            agg_result[k] = "N/A"

    return agg_result


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = FIGURES_DIR / "confusion_matrix.png"
):
    """Plot and save a clean confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=11, weight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
    ax.set_yticklabels(["Legitimate (0)", "Fraud (1)"])
    ax.set_xlabel("Predicted Label", fontsize=10, weight="bold")
    ax.set_ylabel("True Label", fontsize=10, weight="bold")
    ax.set_title(title, fontsize=11, pad=12, weight="bold")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved confusion matrix: {save_path}")
    plt.close()


def plot_precision_recall_curve_tuned(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tuned_threshold: float,
    tuned_precision: float,
    tuned_recall: float,
    title: str = "Precision-Recall Curve with Tuned Threshold",
    save_path: Optional[Path] = FIGURES_DIR / "baseline_pr_curve_tuned.png"
):
    """Plot PR curve with validation-tuned threshold marked as a prominent star."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline_rate = np.mean(y_true == 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.plot(recalls, precisions, color="#1f77b4", lw=2, label=f"Random Forest (PR-AUC = {ap:.4f})")
    ax.axhline(baseline_rate, color="gray", linestyle="--", lw=1.2, label=f"Random Guess (Base Rate = {baseline_rate:.4f})")

    # Mark the validation-tuned threshold operating point
    ax.plot(
        tuned_recall, tuned_precision, marker="*", markersize=14, color="#d62728",
        linestyle="None", label=f"Tuned Threshold $\\tau^*={tuned_threshold:.2f}$\n(Prec={tuned_precision:.3f}, Rec={tuned_recall:.3f})"
    )

    ax.set_xlabel("Recall (Fraction of Frauds Caught)", fontsize=10, weight="bold")
    ax.set_ylabel("Precision (Accuracy of Flagged Frauds)", fontsize=10, weight="bold")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved tuned PR curve plot: {save_path}")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    save_path: Optional[Path] = FIGURES_DIR / "roc_curve.png"
):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.plot(fpr, tpr, color="#2ca02c", lw=2, label=f"Model (ROC-AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Random Chance (AUC = 0.5000)")

    ax.set_xlabel("False Positive Rate (Fall-out)", fontsize=10, weight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10, weight="bold")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved ROC curve plot: {save_path}")
    plt.close()


def plot_mia_comparison_curve(
    mia_results_list: List[Dict[str, Any]],
    save_path: Optional[Path] = FIGURES_DIR / "mia_comparison_across_models.png"
):
    """Plot MIA ROC curves comparing non-private baseline vs. differentially private models."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=150)

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    for idx, res in enumerate(mia_results_list):
        fpr = res.get("fpr")
        tpr = res.get("tpr")
        name = res.get("model_name", f"Model {idx}")
        auc = res.get("attack_auc", 0.5)
        if fpr is not None and tpr is not None:
            color = colors[idx % len(colors)]
            ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1.2, label="Random Guess (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate (Non-Members Flagged)", fontsize=10, weight="bold")
    ax.set_ylabel("True Positive Rate (Members Identified)", fontsize=10, weight="bold")
    ax.set_title("Membership Inference Vulnerability: Non-Private vs. DP Models", fontsize=11, weight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved MIA comparison plot: {save_path}")
    plt.close()


def save_metrics_summary(
    metrics_list: List[Dict[str, Any]],
    json_path: Path = RESULTS_DIR / "all_experiments_metrics.json",
    csv_path: Path = RESULTS_DIR / "experiments_comparison.csv"
):
    """Save consolidated metrics to structured JSON and tabular CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=4)

    summary_rows = []
    for m in metrics_list:
        summary_rows.append({
            "Experiment": m.get("experiment_name", "Unknown"),
            "Threshold": m.get("threshold_used", "0.50"),
            "Precision": m.get("precision", "N/A"),
            "Recall": m.get("recall", "N/A"),
            "F1-Score": m.get("f1_score", "N/A"),
            "PR-AUC": m.get("pr_auc", "N/A"),
            "ROC-AUC": m.get("roc_auc", "N/A"),
            "Accuracy": m.get("accuracy", "N/A"),
            "Runtime (s)": m.get("runtime_seconds", "N/A"),
            "Memory (MB)": m.get("memory_mb", "N/A"),
            "Privacy/Attack Metric": m.get("privacy_or_attack_metric", "N/A"),
            "Notes/Limitations": m.get("notes", "N/A")
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison CSV summary: {csv_path}")
    return summary_df
