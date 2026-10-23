r"""
Module 3: Local Differential Privacy (LDP) for Numerical Features.
Demonstrates client-side feature randomization where the central aggregator is untrusted.

Mathematical Formulation:
- In Local Differential Privacy (LDP), each client perturbs their local value $x \in [a, b]$
  using a randomized mechanism $\mathcal{M}_{loc}$ before transmission:
  $P[\mathcal{M}_{loc}(x) \in S] \le e^{\epsilon_{loc}} P[\mathcal{M}_{loc}(x') \in S]$ for all $x, x' \in [a, b]$.
- Bounded 1D Laplace Mechanism:
  Sensitivity $\Delta = b - a$.
  Noise $w \sim \text{Laplace}(0, \sigma = \Delta / \epsilon_{loc})$.
  Perturbed feature $x_{priv} = x + w$.
- Comparison:
  Central DP injects noise $O(\Delta / \epsilon)$ across the entire dataset.
  LDP injects noise $O(\Delta / \epsilon_{loc})$ on EVERY individual sample, requiring significantly
  higher aggregate noise and thus exhibiting a steeper utility penalty in extreme fraud detection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier

from configs.config import LDP_CONFIG, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED
from src.evaluation import compute_classification_metrics, ResourceTracker

logger = logging.getLogger(__name__)


def apply_laplace_ldp(
    feature_values: np.ndarray,
    epsilon: float,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    clip_output: bool = True,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, Dict[str, Any]]:
    r"""
    Apply Bounded Laplace Local Differential Privacy to a continuous numerical feature.

    Parameters:
    -----------
    feature_values : np.ndarray
        1D array of original numerical values.
    epsilon : float
        Local privacy budget $\epsilon_{loc}$. Smaller values inject more noise.
    lower_bound : float
        Domain minimum $a$. If None, computed from 1st percentile to avoid outlier distortion.
    upper_bound : float
        Domain maximum $b$. If None, computed from 99th percentile.
    clip_output : bool
        Whether to re-clip perturbed values to $[a, b]$.

    Returns:
    --------
    perturbed_values : np.ndarray
    mechanism_metadata : Dict[str, Any]
    """
    if lower_bound is None:
        lower_bound = float(np.percentile(feature_values, 1))
    if upper_bound is None:
        upper_bound = float(np.percentile(feature_values, 99))

    # Clip inputs to enforce strict bounded domain
    clipped_inputs = np.clip(feature_values, lower_bound, upper_bound)
    sensitivity = float(upper_bound - lower_bound)

    scale = sensitivity / epsilon
    rng = np.random.RandomState(random_state)
    noise = rng.laplace(loc=0.0, scale=scale, size=len(feature_values))

    perturbed = clipped_inputs + noise
    if clip_output:
        perturbed = np.clip(perturbed, lower_bound, upper_bound)

    # Compute distortion metrics
    mse = float(np.mean((feature_values - perturbed) ** 2))
    mae = float(np.mean(np.abs(feature_values - perturbed)))
    w_dist = float(wasserstein_distance(feature_values, perturbed))

    meta = {
        "epsilon": epsilon,
        "sensitivity": round(sensitivity, 4),
        "domain_bounds": [round(lower_bound, 4), round(upper_bound, 4)],
        "laplace_scale": round(scale, 4),
        "feature_mse": round(mse, 4),
        "feature_mae": round(mae, 4),
        "wasserstein_distance": round(w_dist, 4)
    }

    return perturbed, meta


def run_ldp_demonstration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_index: int = 0,
    feature_name: str = "Amount",
    epsilons: Optional[List[float]] = None,
    save_outputs: bool = True
) -> Dict[str, Any]:
    r"""
    Execute comprehensive LDP demonstration:
    1. Perturb selected feature across multiple local $\epsilon$ levels.
    2. Measure statistical feature distortion (MSE, Wasserstein distance).
    3. Train downstream Random Forest on perturbed data to evaluate impact on fraud detection.
    4. Generate comparative distribution plots.
    """
    if epsilons is None:
        epsilons = LDP_CONFIG["epsilons"]

    orig_feat = X_train[:, feature_index].copy()
    lower_b = float(np.percentile(orig_feat, 1))
    upper_b = float(np.percentile(orig_feat, 99))

    results_table = []
    perturbed_arrays = {}

    for eps in epsilons:
        pert_train, meta = apply_laplace_ldp(
            orig_feat,
            epsilon=eps,
            lower_bound=lower_b,
            upper_bound=upper_b
        )
        perturbed_arrays[eps] = pert_train

        # Downstream utility check: train RF with perturbed feature
        X_tr_perturbed = X_train.copy()
        X_tr_perturbed[:, feature_index] = pert_train

        # Also perturb test feature to simulate full LDP deployment
        pert_test, _ = apply_laplace_ldp(
            X_test[:, feature_index],
            epsilon=eps,
            lower_bound=lower_b,
            upper_bound=upper_b
        )
        X_te_perturbed = X_test.copy()
        X_te_perturbed[:, feature_index] = pert_test

        with ResourceTracker() as tracker:
            rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
            rf.fit(X_tr_perturbed, y_train)

            y_pred = rf.predict(X_te_perturbed)
            y_prob = rf.predict_proba(X_te_perturbed)[:, 1]

        m = compute_classification_metrics(y_test, y_pred, y_prob, f"LDP_RF_eps_{eps}")
        m.update(tracker.get_summary())
        m.update(meta)
        m["feature_perturbed"] = feature_name
        m["privacy_or_attack_metric"] = f"LDP Laplace $\\epsilon_{{loc}}={eps}$, Wasserstein Dist={meta['wasserstein_distance']}"
        m["notes"] = f"Client-side LDP perturbation applied to feature '{feature_name}'. Aggregator is untrusted."
        results_table.append(m)

    summary = {
        "module": "Module 3: Local Differential Privacy",
        "perturbed_feature": feature_name,
        "mechanism": "Bounded 1D Laplace Mechanism",
        "domain_bounds": [round(lower_b, 4), round(upper_b, 4)],
        "experiments": results_table,
        "analytical_takeaway": (
            "Local Differential Privacy protects each user's raw feature values before transmission "
            "to the central server, eliminating the single point of failure of a trusted aggregator. "
            "However, because noise is added to every record independently, the variance scales as O(1/epsilon) "
            "per record, leading to observable degradation in downstream F1-score when noise is severe."
        )
    }

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        with open(RESULTS_DIR / "ldp_demonstration_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        pd.DataFrame(results_table).to_csv(RESULTS_DIR / "ldp_comparison.csv", index=False)

        # Plot feature distribution before and after perturbation
        fig, axes = plt.subplots(1, len(epsilons) + 1, figsize=(3.8 * (len(epsilons) + 1), 3.5), dpi=150, sharey=True)

        axes[0].hist(orig_feat, bins=30, color="#1f77b4", alpha=0.75, edgecolor="black")
        axes[0].set_title(f"Original '{feature_name}'\n(No Noise)", fontsize=10, weight="bold")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Frequency")

        for idx, eps in enumerate(epsilons):
            ax = axes[idx + 1]
            pert = perturbed_arrays[eps]
            ax.hist(pert, bins=30, color="#ff7f0e", alpha=0.75, edgecolor="black")
            ax.set_title(f"LDP $\\epsilon={eps}$\n(WD={results_table[idx]['wasserstein_distance']:.2f})", fontsize=10, weight="bold")
            ax.set_xlabel("Value")

        plt.suptitle(f"Local Differential Privacy: Feature '{feature_name}' Perturbation Distributions", fontsize=12, weight="bold", y=1.03)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ldp_feature_distributions.png", bbox_inches="tight")
        plt.close()
        logger.info("LDP demonstration completed and artifacts saved.")

    return summary


if __name__ == "__main__":
    from src.data_preprocessing import load_dataset, split_data, preprocess_features
    df, _ = load_dataset()
    train_df, test_df, val_df, _ = split_data(df)
    X_tr, X_te, _, y_tr, y_te, _, feats, _ = preprocess_features(train_df, test_df, val_df)
    feat_idx = feats.index("Amount") if "Amount" in feats else 0
    res = run_ldp_demonstration(X_tr, y_tr, X_te, y_te, feature_index=feat_idx, feature_name="Amount", epsilons=[0.5, 1.0, 5.0])
    print("LDP Summary:", res["analytical_takeaway"])
