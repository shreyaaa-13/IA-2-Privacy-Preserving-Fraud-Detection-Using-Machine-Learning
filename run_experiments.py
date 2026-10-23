"""
Master Experiment Runner for IA 2: Privacy-Preserving Fraud Detection.
Executes the complete benchmark on the real 284,807 Kaggle dataset (or synthetic test harness),
evaluating all five syllabus modules:
- Experiment A: Baseline Random Forest with Validation-Only Threshold Tuning & Multi-Seed Uncertainty
- Experiment B (Module 1): Comparative Membership Inference Attack (Non-Private vs. DP Models)
- Experiment C (Module 2): Central DP Sweep (diffprivlib) & DP-SGD with Opacus (RDP Accounting)
- Experiment D (Module 3): Local Differential Privacy (Bounded Laplace on Amount)
- Experiment E (Module 4): Privacy-Preserving Data Mining (Continuous Microaggregation k=5, 10)
- Experiment F (Module 5): Adversarial Compressive Privacy Autoencoder (Min-Max Optimization)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    REAL_DATASET_PATH,
    SYNTHETIC_DATASET_PATH,
    RESULTS_DIR,
    FIGURES_DIR,
    MULTI_SEEDS,
    ensure_directories
)
from src.data_preprocessing import (
    load_dataset,
    inspect_dataset,
    split_data,
    preprocess_features,
    save_preprocessing_metadata
)
from src.baseline_model import run_baseline_experiment
from src.membership_inference import run_comparative_mia_experiment
from src.differential_privacy import run_dp_privacy_utility_sweep, train_dp_sgd_opacus
from src.local_differential_privacy import run_ldp_demonstration
from src.privacy_preserving_mining import run_ppdm_demonstration
from src.compressive_privacy import run_compressive_privacy_demonstration
from src.evaluation import save_metrics_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="IA 2: Privacy-Preserving Fraud Detection Complete Pipeline")
    parser.add_argument(
        "--mode",
        choices=["auto", "test", "real"],
        default="auto",
        help="Dataset mode: 'real' (Kaggle 284k dataset), 'test' (synthetic mock data), or 'auto' (detects real, falls back to test)."
    )
    parser.add_argument(
        "--split",
        choices=["stratified", "chronological"],
        default="stratified",
        help="Partitioning strategy: 'stratified' (recommended) or 'chronological'."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast sweep configuration for rapid validation."
    )
    args = parser.parse_args()

    ensure_directories()
    start_total_time = time.perf_counter()

    print("=" * 85)
    print("SEMESTER VII: SECURITY AND PRIVACY FOR MACHINE LEARNING")
    print("IA 2: Implementation, Evaluation and Technical Outcomes")
    print("Project: Privacy-Preserving Fraud Detection Using Machine Learning")
    print(f"Student: Shreya Jadhav (Roll: 16014223077, Batch: B3)")
    print(f"Execution Mode: {args.mode.upper()} | Split Protocol: {args.split.upper()}")
    print("=" * 85)

    # 1. Dataset Acquisition & Diagnostic Inspection
    target_path = None
    if args.mode == "real":
        if not REAL_DATASET_PATH.exists():
            logger.error(f"Real dataset not found at: {REAL_DATASET_PATH}")
            sys.exit(1)
        target_path = REAL_DATASET_PATH
    elif args.mode == "test":
        target_path = SYNTHETIC_DATASET_PATH

    df, provenance_tag = load_dataset(target_path)
    inspection = inspect_dataset(df)

    print(f"\n[DATASET VERIFIED] Source: {provenance_tag}")
    print(f"Total Transactions: {inspection['total_records']:,}")
    print(f"Fraudulent Transactions: {inspection['class_distribution']['fraudulent_1']} ({inspection['class_distribution']['fraud_prevalence_percentage']}%)")
    print(f"Legitimate Transactions: {inspection['class_distribution']['legitimate_0']:,}")
    print(f"Missing Values: {inspection['total_missing_values']} | Duplicates: {inspection['duplicate_records']}")

    # 2. Split and Preprocess
    train_df, test_df, val_df, split_meta = split_data(df, method=args.split, random_state=MULTI_SEEDS[0])
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names, scaler = preprocess_features(
        train_df, test_df, val_df
    )

    metadata_bundle = {
        "provenance_tag": provenance_tag,
        "inspection": inspection,
        "split_metadata": split_meta,
        "num_features": len(feature_names),
        "features": feature_names
    }
    save_preprocessing_metadata(metadata_bundle)

    all_metrics = []

    # 3. Experiment A: Baseline Fraud Detector (Default vs. Tuned Threshold + Multi-Seed)
    print("\n" + "-" * 85)
    print("EXPERIMENT A: Baseline Random Forest & Validation-Only Threshold Tuning")
    print("-" * 85)
    seeds_to_run = [MULTI_SEEDS[0]] if args.fast else MULTI_SEEDS
    detector, m_default, m_tuned, m_multiseed = run_baseline_experiment(
        df=df, split_method=args.split, seeds=seeds_to_run
    )
    all_metrics.append(m_default)
    all_metrics.append(m_tuned)
    all_metrics.append(m_multiseed)

    print(f"Default Threshold (0.50): Prec={m_default['precision']:.4f} | Rec={m_default['recall']:.4f} | F1={m_default['f1_score']:.4f} | PR-AUC={m_default['pr_auc']:.4f}")
    print(f"Tuned Threshold  ({m_tuned['threshold_used']:.2f}): Prec={m_tuned['precision']:.4f} | Rec={m_tuned['recall']:.4f} | F1={m_tuned['f1_score']:.4f} | PR-AUC={m_tuned['pr_auc']:.4f}")
    print(f"Multi-Seed Tuned ({len(seeds_to_run)} seeds): F1 = {m_multiseed.get('f1_score', 'N/A')}")

    # 4. Experiment B (Module 1): Comparative Membership Inference Attack
    print("\n" + "-" * 85)
    print("EXPERIMENT B (MODULE 1): Comparative Membership Inference Attack (Non-Private vs. DP)")
    print("-" * 85)
    # If running on full 284k dataset, subsample 10,000 records for fast shadow MIA to bound execution time
    sample_size_mia = min(len(X_train), 10000)
    idx_mia = np.random.RandomState(MULTI_SEEDS[0]).choice(len(X_train), sample_size_mia, replace=False)
    mia_out = run_comparative_mia_experiment(X_train[idx_mia], y_train[idx_mia])
    for r in mia_out["raw_results"]:
        all_metrics.append({
            "experiment_name": f"MIA_vs_{r['model_name']}",
            "threshold_used": "N/A",
            "precision": r["attack_precision"],
            "recall": r["attack_recall"],
            "f1_score": r["attack_f1"],
            "pr_auc": "N/A",
            "roc_auc": r["attack_auc"],
            "accuracy": r["attack_accuracy"],
            "privacy_or_attack_metric": f"MIA AUC: {r['attack_auc']:.4f} | Advantage: {r['attacker_advantage']:.4f}",
            "notes": f"Shadow-model MIA evaluated against {r['model_name']}."
        })
    print(mia_out["comparison_table"][["model_evaluated", "attack_auc", "attack_accuracy", "attacker_advantage"]].to_string(index=False))

    # 5. Experiment C (Module 2): Central Differential Privacy & DP-SGD
    print("\n" + "-" * 85)
    print("EXPERIMENT C (MODULE 2): Central Differential Privacy Sweep & Opacus DP-SGD")
    print("-" * 85)
    dp_epsilons = [0.1, 1.0, 10.0] if args.fast else [0.1, 0.5, 1.0, 5.0, 10.0]
    dp_sweep_df, dp_summary = run_dp_privacy_utility_sweep(
        X_train, y_train, X_test, y_test, epsilons=dp_epsilons
    )
    for dp_rec in dp_summary["dp_sweep_results"]:
        all_metrics.append(dp_rec)
    print("diffprivlib Sweep:")
    print(dp_sweep_df[["experiment_name", "epsilon", "precision", "recall", "f1_score", "pr_auc"]].to_string(index=False))

    # Opacus DP-SGD
    print("\nTraining DP-SGD Neural Network with Opacus (RDP Accounting)...")
    dp_sgd_m, ref_nn_m = train_dp_sgd_opacus(
        X_train, y_train, X_test, y_test,
        target_epsilon=5.0,
        epochs=3 if args.fast else 5,
        batch_size=512
    )
    all_metrics.append(ref_nn_m)
    all_metrics.append(dp_sgd_m)
    print(f"Non-Private NN Ref : Prec={ref_nn_m['precision']:.4f} | Rec={ref_nn_m['recall']:.4f} | F1={ref_nn_m['f1_score']:.4f} | PR-AUC={ref_nn_m['pr_auc']:.4f}")
    print(f"Opacus DP-SGD (eps={dp_sgd_m['achieved_epsilon']:.2f}): Prec={dp_sgd_m['precision']:.4f} | Rec={dp_sgd_m['recall']:.4f} | F1={dp_sgd_m['f1_score']:.4f} | PR-AUC={dp_sgd_m['pr_auc']:.4f}")

    # 6. Experiment D (Module 3): Local Differential Privacy
    print("\n" + "-" * 85)
    print("EXPERIMENT D (MODULE 3): Local Differential Privacy (Bounded Laplace on Amount)")
    print("-" * 85)
    amt_idx = feature_names.index("Amount") if "Amount" in feature_names else 0
    ldp_summary = run_ldp_demonstration(
        X_train, y_train, X_test, y_test,
        feature_index=amt_idx,
        feature_name="Amount",
        epsilons=[0.5, 1.0, 5.0]
    )
    for ldp_rec in ldp_summary["experiments"]:
        all_metrics.append(ldp_rec)
    for exp in ldp_summary["experiments"]:
        print(f"  eps={exp['epsilon']}: F1={exp['f1_score']:.4f} | PR-AUC={exp['pr_auc']:.4f} | Wasserstein Dist={exp['wasserstein_distance']:.4f}")

    # 7. Experiment E (Module 4): Privacy-Preserving Data Mining (Microaggregation)
    print("\n" + "-" * 85)
    print("EXPERIMENT E (MODULE 4): Privacy-Preserving Data Mining (Continuous Microaggregation)")
    print("-" * 85)
    # Use subsample for k-means microaggregation if dataset > 50,000 to keep runtime bounded
    n_ppdm = min(len(X_train), 25000)
    idx_ppdm = np.random.RandomState(MULTI_SEEDS[0]).choice(len(X_train), n_ppdm, replace=False)
    n_ppdm_te = min(len(X_test), 5000)
    idx_ppdm_te = np.random.RandomState(MULTI_SEEDS[0]).choice(len(X_test), n_ppdm_te, replace=False)

    ppdm_summary = run_ppdm_demonstration(
        X_train[idx_ppdm], y_train[idx_ppdm], X_test[idx_ppdm_te], y_test[idx_ppdm_te],
        k_values=[5, 10]
    )
    for ppdm_rec in ppdm_summary["experiments"]:
        all_metrics.append(ppdm_rec)
    for exp in ppdm_summary["experiments"]:
        print(f"  k={exp.get('k_anonymity')}: F1={exp['f1_score']:.4f} | Info Loss={exp.get('information_loss_pct')}%")

    # 8. Experiment F (Module 5): Adversarial Compressive Privacy Autoencoder
    print("\n" + "-" * 85)
    print("EXPERIMENT F (MODULE 5): Adversarial Compressive Privacy Autoencoder")
    print("-" * 85)
    comp_summary = run_compressive_privacy_demonstration(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names,
        latent_dimensions=[4, 8, 16] if not args.fast else [4, 8]
    )
    for comp_rec in comp_summary["experiments"]:
        all_metrics.append(comp_rec)
    for exp in comp_summary["experiments"]:
        print(f"  Latent Dim={exp.get('latent_dimension')}, lambda={exp.get('lambda_adversary')}: Task F1={exp['f1_score']:.4f} | Adv AUC(S)={exp.get('adversary_sensitive_auc')}")

    # 9. Consolidated Table and Wrap-up
    summary_df = save_metrics_summary(all_metrics)
    total_elapsed = time.perf_counter() - start_total_time

    print("\n" + "=" * 85)
    print(f"ALL IA 2 EXPERIMENTS COMPLETED IN {total_elapsed:.1f} SECONDS")
    print(f"Visual Figures: {FIGURES_DIR}")
    print(f"Quantitative Results: {RESULTS_DIR}")
    print(f"Consolidated CSV: {RESULTS_DIR / 'experiments_comparison.csv'}")
    print("=" * 85)


if __name__ == "__main__":
    main()
