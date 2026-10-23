# Visual Figures & Diagnostic Plots Directory

This directory stores publication-quality diagnostic plots generated during model training and evaluation.

---

## Generated Diagnostic Plots

1. **`baseline_confusion_matrix_stratified.png`**:
   - Heatmap displaying True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP) for the baseline Random Forest classifier.
2. **`baseline_pr_curve_stratified.png`**:
   - Precision-Recall curve plotting precision against recall across all classification thresholds, including the baseline random-chance horizontal line.
3. **`baseline_roc_curve_stratified.png`**:
   - Receiver Operating Characteristic (ROC) curve plotting True Positive Rate (Sensitivity) vs. False Positive Rate.
4. **`mia_roc_curve.png`**:
   - ROC curve for the Membership Inference Attacker evaluating classification of training members vs. unseen holdout samples against the diagonal 0.50 random-guess line.
5. **`dp_privacy_utility_tradeoff.png`**:
   - Privacy-Utility Pareto frontier plotting F1-Score and PR-AUC against Differential Privacy budget $\epsilon$ (log scale).
6. **`ldp_feature_distributions.png`**:
   - Side-by-side histogram plots showing the distribution of the transaction `Amount` feature before and after Local Differential Privacy Laplace perturbation across varying $\epsilon_{loc}$.
