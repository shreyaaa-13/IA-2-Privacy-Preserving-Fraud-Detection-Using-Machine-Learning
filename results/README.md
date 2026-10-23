# Experimental Results Directory

This directory stores quantitative evaluation artifacts produced by the experimental pipeline.

---

## 1. Summary Artifacts

- **`experiments_comparison.csv`**: Consolidated comparison table of all experiments across all 5 syllabus modules, displaying Precision, Recall, F1-Score, PR-AUC, ROC-AUC, Accuracy, Privacy/Attack Metric, and Notes/Limitations.
- **`all_experiments_metrics.json`**: Machine-readable JSON dictionary containing complete detailed performance metrics and metadata.
- **`preprocessing_metadata.json`**: Records preprocessing parameters, dataset provenance tag, class distribution counts, missing values, duplicates, and feature lists.

---

## 2. Module-Specific Artifacts

- **Baseline Fraud Model (Experiment A):**
  - `baseline_metrics_stratified.json`: Performance metrics of the non-private Random Forest classifier under stratified data partitioning.
- **Membership Inference Attack (Module 1 / Experiment B):**
  - `membership_inference_metrics.json`: Attacker ROC-AUC, Balanced Accuracy, Attacker Advantage over 50% random chance, and Confusion Matrix.
- **Differential Privacy Sweep (Module 2 / Experiment C):**
  - `differential_privacy_results.json`: Full results for IBM `diffprivlib` DP-Logistic Regression across $\epsilon \in [0.1, 0.5, 1.0, 5.0, 10.0]$.
  - `dp_epsilon_sweep_comparison.csv`: Tabular sweep comparison.
- **Local Differential Privacy (Module 3):**
  - `ldp_demonstration_results.json`: Feature distortion metrics (MSE, Wasserstein distance) and downstream fraud detection performance.
  - `ldp_comparison.csv`: Tabular comparison across local $\epsilon$.
- **Privacy-Preserving Data Mining (Module 4):**
  - `ppdm_demonstration_results.json`: Continuous microaggregation for $k \in \{5, 10\}$ with Information Loss percentages and PCA vs DP analysis.
  - `ppdm_comparison.csv`: Tabular comparison.
- **Compressive Privacy (Module 5):**
  - `compressive_privacy_results.json`: Subspace representation compression results, sensitive attribute leakage AUC, and literature citations.
  - `compressive_privacy_comparison.csv`: Tabular comparison across latent dimensions.
