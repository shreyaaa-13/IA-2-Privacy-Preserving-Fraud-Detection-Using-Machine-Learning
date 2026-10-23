# Syllabus Module Mapping and Academic Assessment Rubric

**Subject:** Security and Privacy for Machine Learning (Semester VII)  
**Assessment:** IA 1 — Problem Formulation and Preliminary Implementation (20 Marks)  
**Project Title:** Privacy-Preserving Fraud Detection Using Machine Learning  
**Candidate Name:** `[STUDENT_NAME_PLACEHOLDER]` | **Roll Number:** `[ROLL_NO_PLACEHOLDER]`  
**Division / Batch:** `[DIVISION_BATCH_PLACEHOLDER]` | **Course Faculty:** `[FACULTY_NAME_PLACEHOLDER]`  

---

## 1. Syllabus Module Mapping

This project explicitly addresses all five prescribed syllabus modules for **Security and Privacy for Machine Learning**, establishing a clear separation between primary empirical implementations and analytical demonstrations.

| Module No. | Syllabus Module Title | Core Theoretical Concepts Addressed | Implementation / Analytical Artifact | Primary Source Code | Rubric Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Module 1** | **Privacy Considerations in ML** | Privacy threat models (black-box vs. white-box), training data leakage, Membership Inference Attacks (MIA), shadow model training, entropy/loss feature extraction. | **Implemented & Evaluated:** Controlled MIA experiment against Random Forest baseline using 4-way disjoint partitioning, shadow modeling, and attack evaluation against a 50% random chance baseline. | [`src/membership_inference.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/membership_inference.py) | Rubric 1, 2, 4, 5 (4 Marks) |
| **Module 2** | **Differential Privacy for ML** | $(\epsilon, \delta)$-Differential Privacy, pure $\epsilon$-DP, global sensitivity $\Delta f$, $L_2$ gradient/feature clipping ($C$), Laplace & Gaussian mechanisms, objective perturbation, Privacy-Utility Trade-off. | **Implemented & Evaluated:** IBM `diffprivlib` pure $\epsilon$-DP Logistic Regression sweep over $\epsilon \in [0.1, 0.5, 1.0, 5.0, 10.0]$ with sensitivity bound $C=5.0$; Pareto frontier analysis against non-private baseline. | [`src/differential_privacy.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/differential_privacy.py) | Rubric 1, 2, 4, 5 (4 Marks) |
| **Module 3** | **Local Differential Privacy for ML** | Untrusted aggregator threat model, client-side randomization, randomized response, 1D bounded Laplace mechanism, Wasserstein distance, noise accumulation ($O(1/\epsilon_{loc})$ per record). | **Implemented Demonstration:** Client-side bounded Laplace perturbation on transaction `Amount` feature across varying local $\epsilon$; quantitative measurement of feature distortion (MSE, Wasserstein distance) and downstream fraud detection impact. | [`src/local_differential_privacy.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/local_differential_privacy.py) | Rubric 1, 2, 4 (4 Marks) |
| **Module 4** | **Privacy-Preserving Data Mining (PPDM)** | Quasi-identifiers, $k$-anonymity, $l$-diversity, $t$-closeness, continuous microaggregation, centroid replacement, information loss quantification (SSE/SST), PCA vs. DP formal distinction. | **Implemented Demonstration & Analytical Critique:** Continuous microaggregation for $k \in \{5, 10\}$ with Information Loss calculation; rigorous technical explanation of why PCA is linear dimensionality reduction, NOT a differential privacy guarantee. | [`src/privacy_preserving_mining.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/privacy_preserving_mining.py) | Rubric 1, 2, 4 (4 Marks) |
| **Module 5** | **Compressive Privacy for ML** | Shannon Mutual Information, Information Bottleneck principle $\min [I(Z; S) - \beta I(Z; Y)]$, representation compression, sensitive attribute suppression vs. task utility preservation. | **Analytical Demonstration & Literature Synthesis:** Empirical latent representation sweep analyzing downstream fraud utility vs. adversary leakage of sensitive proxy attribute $S$; literature synthesis of Kung (2017) and Calmon et al. (2017). Full adversarial autoencoder planned for IA 2. | [`src/compressive_privacy.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/compressive_privacy.py) | Rubric 1, 2, 4 (4 Marks) |

---

## 2. Assessment Rubric Breakdown (20 Marks Total)

The project structure directly aligns with the college evaluation rubric:

### Rubric 1: Problem Definition and Module Mapping (4 Marks)
- **Problem Formulation:** Formalizes the severe class imbalance problem in credit card fraud (~0.17% fraud prevalence) and explains why conventional accuracy is deceptive.
- **Privacy Threat Modeling:** Establishes the attacker goal in Membership Inference ($m \in \{0, 1\}$) versus the defender goal in fraud classification ($y \in \{0, 1\}$).
- **Comprehensive Mapping:** Systematically links all five syllabus modules to reproducible source code modules, configurations, and evaluation artifacts.

### Rubric 2: Conceptual Understanding (4 Marks)
- **Differential Privacy Foundations:** Explains pure $\epsilon$-DP, $(\epsilon, \delta)$-DP, bounded clipping ($C$), and calibrated noise perturbation.
- **Local vs. Central DP:** Clarifies the trust model differences between client-side randomization (untrusted server) and central aggregation.
- **PPDM & PCA Distinction:** Defends why PCA-transformed features do not provide mathematical differential privacy guarantees.
- **Compressive Privacy Theory:** Formulates the Information Bottleneck optimization problem for privacy-preserving representation learning.

### Rubric 3: Dataset Identification and Preliminary Preparation (4 Marks)
- **Authentic Provenance:** Complete documentation of the ULB Kaggle Credit Card Fraud dataset (284,807 records, 492 frauds, ODbL license, Andrea Dal Pozzolo et al.).
- **Diagnostic Inspection:** Detailed inspection functions for missing values, duplicates, feature data types, and severe imbalance ratios.
- **Anti-Leakage Data Splitting:** Stratified split preserving the 0.17% fraud prevalence across train, validation, and test sets.
- **Leak-Free Scaling:** `RobustScaler` fitted strictly on training data, transforming validation and test partitions without data contamination.
- **Synthetic Verification Harness:** Provides a deterministic synthetic dataset generator matching the ULB schema so unit tests and pipelines execute offline out-of-the-box.

### Rubric 4: Initial Implementation or Demonstration (4 Marks)
- **Baseline Fraud Model:** Random Forest classifier with `class_weight='balanced'`, evaluated across Precision, Recall, F1, PR-AUC, and Confusion Matrix.
- **Controlled MIA Attack:** Shadow model and attack classifier trained on 4-way disjoint data partitions without evaluation leakage.
- **Mathematically Grounded DP:** IBM `diffprivlib` DP-Logistic Regression parameter sweep over $\epsilon \in [0.1, 0.5, 1.0, 5.0, 10.0]$.
- **Modular Architecture:** Beginner-friendly, modular Python source files with central configuration, CLI runner, and Jupyter Notebook.

### Rubric 5: Preliminary Evaluation and Documentation (4 Marks)
- **Authoritative Metrics:** Rejects accuracy; reports Precision-Recall AUC (PR-AUC), Sensitivity (Recall), F1-score, and Attack ROC-AUC.
- **Pareto Trade-Off Curve:** Visualizes empirical Privacy-Utility frontiers across privacy budgets $\epsilon$.
- **Complete Artifacts:** Persists structured JSON metrics, comparison CSV tables, and publication-quality diagnostic plots.
- **Academic Documentation:** Produces a preliminary IA 1 Markdown report, beginner guide, and viva defense questionnaire.
