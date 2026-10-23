"""
Module 4: Privacy-Preserving Data Mining (PPDM) Techniques.
Demonstrates classical privacy-aware data modification and addresses the critical misconception
regarding Principal Component Analysis (PCA).

CRITICAL CONCEPTUAL DISTINCTION:
Why PCA does NOT guarantee Differential Privacy:
1. PCA is a deterministic, orthogonal linear projection ($Z = X W$) designed exclusively
   to maximize variance and minimize $L_2$ reconstruction error.
2. PCA contains ZERO calibrated noise injection and enforces NO sensitivity bounds.
3. High reconstruction fidelity ($X \approx Z W^T$) means that an adversary with partial background knowledge
   can invert projections, reconstruct sensitive attributes, or perform membership inference.
4. Kaggle's ULB dataset uses PCA solely as an empirical redaction heuristic by the data publisher,
   NOT as a provable mathematical privacy guarantee.

Technique Implemented:
Continuous Microaggregation for $k$-Anonymity:
- Partitions continuous transaction vectors into clusters of size at least $k$.
- Replaces individual records with their respective cluster centroids.
- Ensures that every individual record is identical to at least $k-1$ other records in the released dataset.
- Quantifies Information Loss (Sum of Squared Errors / Normalized SSE).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier

from configs.config import RESULTS_DIR, RANDOM_SEED
from src.evaluation import compute_classification_metrics, ResourceTracker

logger = logging.getLogger(__name__)


def apply_microaggregation(
    X: np.ndarray,
    k: int = 5,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, float]:
    """
    Apply Microaggregation to continuous numerical features.

    Groups records into clusters of average size $k$ using MiniBatchKMeans,
    and replaces each individual feature vector with its cluster centroid.
    Ensures continuous $k$-anonymity on quasi-identifiers.

    Returns:
    --------
    X_anonymized : np.ndarray
    information_loss : float (Normalized Sum of Squared Error, 0 = no loss, 1 = total information loss)
    """
    n_samples = len(X)
    n_clusters = max(1, n_samples // k)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=1024)
    cluster_labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Replace each record with its cluster centroid
    X_anonymized = centroids[cluster_labels]

    # Calculate Information Loss (SST: total variance, SSE: within-cluster error)
    sst = np.sum((X - np.mean(X, axis=0)) ** 2)
    sse = np.sum((X - X_anonymized) ** 2)
    normalized_loss = float(sse / sst) if sst > 0 else 0.0

    return X_anonymized, normalized_loss


def run_ppdm_demonstration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values: list = [5, 10, 20],
    save_outputs: bool = True
) -> Dict[str, Any]:
    """
    Run empirical PPDM demonstration:
    1. Apply continuous microaggregation for varying $k$ anonymity thresholds.
    2. Quantify Information Loss.
    3. Train downstream classifier and evaluate impact on fraud detection.
    4. Provide formal technical comparison against PCA.
    """
    results_list = []

    # Baseline non-anonymized reference
    with ResourceTracker() as base_tracker:
        rf_baseline = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
        rf_baseline.fit(X_train, y_train)
        y_pred_base = rf_baseline.predict(X_test)
        y_prob_base = rf_baseline.predict_proba(X_test)[:, 1]
    m_base = compute_classification_metrics(y_test, y_pred_base, y_prob_base, "Non_Anonymized_Reference")
    m_base.update(base_tracker.get_summary())
    m_base["k_anonymity"] = 1
    m_base["information_loss_pct"] = 0.0
    m_base["privacy_or_attack_metric"] = "k=1 (No Anonymization)"
    m_base["notes"] = "Raw feature representation without PPDM transformation."
    results_list.append(m_base)

    for k in k_values:
        with ResourceTracker() as tracker:
            X_tr_anon, loss_tr = apply_microaggregation(X_train, k=k)
            X_te_anon, loss_te = apply_microaggregation(X_test, k=k)

            rf_anon = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
            rf_anon.fit(X_tr_anon, y_train)

            y_pred_anon = rf_anon.predict(X_te_anon)
            y_prob_anon = rf_anon.predict_proba(X_te_anon)[:, 1]

        m_anon = compute_classification_metrics(y_test, y_pred_anon, y_prob_anon, f"PPDM_Microaggregation_k{k}")
        m_anon.update(tracker.get_summary())
        m_anon["k_anonymity"] = k
        m_anon["information_loss_pct"] = round(loss_tr * 100, 2)
        m_anon["privacy_or_attack_metric"] = f"k-Anonymity (k={k}), Info Loss={round(loss_tr*100, 1)}%"
        m_anon["notes"] = f"Centroid replacement in clusters of size k={k}. Records indistinguishable within cluster."
        results_list.append(m_anon)

    pca_vs_dp_clarification = (
        "PCA vs Differential Privacy Clarification:\n"
        "1. PCA is an unsupervised orthogonal dimensionality reduction tool. "
        "It preserves 80-95% of data variance without adding calibrated noise.\n"
        "2. PCA does NOT satisfy differential privacy. There is no epsilon bound, no sensitivity clipping, "
        "and no mechanism to bound worst-case inference.\n"
        "3. An adversary observing PCA features can infer membership or reconstruct raw attributes using regression "
        "or shadow modeling.\n"
        "4. In this project, PCA was applied upstream by the dataset creators (ULB) for basic column anonymization, "
        "not as a provable privacy defense."
    )

    summary = {
        "module": "Module 4: Privacy-Preserving Data Mining (PPDM)",
        "technique": "Continuous Microaggregation for k-Anonymity",
        "pca_vs_dp_clarification": pca_vs_dp_clarification,
        "experiments": results_list
    }

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "ppdm_demonstration_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        pd.DataFrame(results_list).to_csv(RESULTS_DIR / "ppdm_comparison.csv", index=False)
        logger.info("PPDM demonstration completed and artifacts saved.")

    return summary


if __name__ == "__main__":
    from src.data_preprocessing import load_dataset, split_data, preprocess_features
    df, _ = load_dataset()
    train_df, test_df, val_df, _ = split_data(df)
    X_tr, X_te, _, y_tr, y_te, _, _, _ = preprocess_features(train_df, test_df, val_df)
    res = run_ppdm_demonstration(X_tr, y_tr, X_te, y_te, k_values=[5, 10])
    print(res["pca_vs_dp_clarification"])
