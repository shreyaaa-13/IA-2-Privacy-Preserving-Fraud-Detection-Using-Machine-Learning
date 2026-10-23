"""
Experiment A: Enhanced Baseline Fraud Detection with Validation-Only Threshold Tuning.
Trains a non-private Random Forest Classifier on preprocessed transaction features,
tunes probability decision thresholds strictly on the validation split,
evaluates on the unseen test set, and repeats across multiple seeds for uncertainty quantification.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from configs.config import (
    BASELINE_RF_PARAMS,
    RESULTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    MULTI_SEEDS
)
from src.data_preprocessing import (
    load_dataset,
    inspect_dataset,
    split_data,
    preprocess_features,
    save_preprocessing_metadata
)
from src.evaluation import (
    compute_classification_metrics,
    find_optimal_threshold,
    aggregate_multiseed_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve_tuned,
    plot_roc_curve,
    ResourceTracker
)

logger = logging.getLogger(__name__)


class BaselineFraudDetector:
    """Ensemble Random Forest fraud detector with balanced class weighting."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or BASELINE_RF_PARAMS.copy()
        self.model = RandomForestClassifier(**self.params)
        self.is_fitted = False
        self.feature_names = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[list] = None):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        if feature_names:
            self.feature_names = feature_names
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict() is called.")
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba() is called.")
        return self.model.predict_proba(X)


def run_baseline_experiment(
    df: Optional[pd.DataFrame] = None,
    split_method: str = "stratified",
    seeds: Optional[List[int]] = None,
    save_outputs: bool = True
) -> Tuple[BaselineFraudDetector, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Execute Experiment A with validation threshold tuning and multi-seed evaluation:
    1. Fits Random Forest on X_train
    2. Tunes decision threshold tau* on X_val to maximize F1-score
    3. Evaluates on X_test at default tau=0.50 AND optimal tau*
    4. Evaluates across multiple seeds to compute mean ± std
    """
    if df is None:
        df, tag = load_dataset()
    else:
        tag = "IN-MEMORY-DATAFRAME"

    if seeds is None:
        seeds = MULTI_SEEDS

    inspection = inspect_dataset(df)
    train_df, test_df, val_df, split_meta = split_data(df, method=split_method, random_state=seeds[0])
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names, scaler = preprocess_features(
        train_df, test_df, val_df
    )

    metadata = {
        "dataset_tag": tag,
        "inspection": inspection,
        "split_metadata": split_meta,
        "num_features": len(feature_names),
        "feature_names": feature_names
    }
    if save_outputs:
        save_preprocessing_metadata(metadata)

    # Primary Seed Run (Seed 42)
    with ResourceTracker() as tracker:
        detector = BaselineFraudDetector()
        detector.fit(X_train, y_train, feature_names=feature_names)

        # Probabilities
        y_prob_val = detector.predict_proba(X_val)[:, 1]
        y_prob_test = detector.predict_proba(X_test)[:, 1]

        # Tune threshold strictly on validation set
        best_thresh, val_f1, tuning_meta = find_optimal_threshold(y_val, y_prob_val, metric="f1")

        # Predictions on unseen test set
        y_pred_default = (y_prob_test >= 0.50).astype(int)
        y_pred_tuned = (y_prob_test >= best_thresh).astype(int)

    resource_stats = tracker.get_summary()

    # Metrics at default threshold 0.50
    metrics_default = compute_classification_metrics(
        y_test, y_pred_default, y_prob_test,
        f"Baseline_RF_DefaultThresh_{split_method}",
        threshold_used=0.50
    )
    metrics_default.update(resource_stats)
    metrics_default["notes"] = "Random Forest with default decision threshold tau = 0.50."
    metrics_default["privacy_or_attack_metric"] = "None (Non-Private Baseline)"

    # Metrics at tuned threshold tau*
    metrics_tuned = compute_classification_metrics(
        y_test, y_pred_tuned, y_prob_test,
        f"Baseline_RF_TunedThresh_{split_method}",
        threshold_used=best_thresh
    )
    metrics_tuned.update(resource_stats)
    metrics_tuned.update(tuning_meta)
    metrics_tuned["notes"] = f"Random Forest with validation-tuned threshold tau* = {best_thresh:.4f}."
    metrics_tuned["privacy_or_attack_metric"] = "None (Non-Private Baseline)"

    logger.info(
        f"[Baseline Tuned (Seed 42)] Threshold: {best_thresh:.4f} | Prec: {metrics_tuned['precision']} | "
        f"Rec: {metrics_tuned['recall']} | F1: {metrics_tuned['f1_score']} | PR-AUC: {metrics_tuned['pr_auc']}"
    )

    # Multi-Seed Evaluation for Uncertainty Estimation
    multiseed_records = [metrics_tuned]
    for s in seeds[1:]:
        train_s, test_s, val_s, _ = split_data(df, method=split_method, random_state=s)
        X_tr_s, X_te_s, X_v_s, y_tr_s, y_te_s, y_v_s, _, _ = preprocess_features(train_s, test_s, val_s)

        rf_s = BaselineFraudDetector({"n_estimators": 50, "max_depth": 10, "class_weight": "balanced", "random_state": s, "n_jobs": -1})
        rf_s.fit(X_tr_s, y_tr_s)
        p_val_s = rf_s.predict_proba(X_v_s)[:, 1]
        p_test_s = rf_s.predict_proba(X_te_s)[:, 1]

        best_t_s, _, _ = find_optimal_threshold(y_v_s, p_val_s, metric="f1")
        pred_tuned_s = (p_test_s >= best_t_s).astype(int)
        m_s = compute_classification_metrics(y_te_s, pred_tuned_s, p_test_s, f"Baseline_RF_Seed_{s}", threshold_used=best_t_s)
        multiseed_records.append(m_s)

    multiseed_summary = aggregate_multiseed_metrics(multiseed_records, "Baseline_RF_TunedThresh_MultiSeed")
    multiseed_summary["notes"] = f"Averaged across {len(seeds)} independent random seeds: {seeds}."

    if save_outputs:
        plot_confusion_matrix(
            y_test, y_pred_tuned,
            title=f"Baseline Tuned Confusion Matrix ($\\tau^*={best_thresh:.2f}$)",
            save_path=FIGURES_DIR / f"baseline_confusion_matrix_{split_method}.png"
        )
        plot_precision_recall_curve_tuned(
            y_test, y_prob_test,
            tuned_threshold=best_thresh,
            tuned_precision=metrics_tuned["precision"],
            tuned_recall=metrics_tuned["recall"],
            title=f"Baseline Precision-Recall Curve ($\\tau^*={best_thresh:.2f}$)",
            save_path=FIGURES_DIR / f"baseline_pr_curve_{split_method}.png"
        )
        plot_roc_curve(
            y_test, y_prob_test,
            title=f"Baseline ROC Curve ({split_method.capitalize()})",
            save_path=FIGURES_DIR / f"baseline_roc_curve_{split_method}.png"
        )

        with open(RESULTS_DIR / "baseline_metrics_tuned.json", "w", encoding="utf-8") as f:
            json.dump({"default": metrics_default, "tuned": metrics_tuned, "multiseed": multiseed_summary}, f, indent=4)

    dataset_bundle = {
        "X_train": X_train, "X_test": X_test, "X_val": X_val,
        "y_train": y_train, "y_test": y_test, "y_val": y_val,
        "feature_names": feature_names, "scaler": scaler
    }

    return detector, metrics_default, metrics_tuned, multiseed_summary


if __name__ == "__main__":
    detector, m_def, m_tuned, m_multi = run_baseline_experiment(split_method="stratified")
    print("\n--- Baseline Results (Default vs Tuned) ---")
    print(f"Default Thresh (0.50): Prec={m_def['precision']}, Rec={m_def['recall']}, F1={m_def['f1_score']}")
    print(f"Tuned Thresh ({m_tuned['threshold_used']}): Prec={m_tuned['precision']}, Rec={m_tuned['recall']}, F1={m_tuned['f1_score']}")
    print("Multi-seed:", m_multi["f1_score"])
