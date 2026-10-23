# Final Module Mapping and Syllabus Integration (IA 2)

**Subject:** Security and Privacy for Machine Learning (Semester VII)  
**Assessment:** IA 2 — Implementation, Evaluation and Technical Outcomes (20 Marks)  
**Project Title:** Privacy-Preserving Fraud Detection Using Machine Learning  
**Student Name:** Shreya Jadhav | **Roll Number:** 16014223077 | **Batch:** B3  
**Programme:** Honours in Applied Cyber Security  

---

## 1. Comprehensive Syllabus Module Integration Table

This table maps each of the five syllabus modules to its theoretical concepts, industrial relevance, implementation methodology, code location, quantitative evaluation metrics, verified real-dataset results, and operational limitations.

| Module | Core Concept | Industrial & Security Relevance | Implementation / Empirical Treatment | Primary Code File | Quantitative Evaluation Metric | Verified Real-Dataset Evidence | Operational Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Module 1: Privacy Considerations in ML** | Threat modeling (black-box vs. white-box), training data memorization, shadow-model Membership Inference Attacks (MIA). | Banks deploying fraud detection APIs risk exposing whether a specific cardholder's transaction was part of the training set through probability confidence leakage. | **Implemented & Evaluated:** Controlled 4-way disjoint shadow-model MIA evaluated against both non-private baseline and differentially private models across $\epsilon \in \{0.1, 1.0, 10.0\}$. | [`src/membership_inference.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/membership_inference.py) | Attacker ROC-AUC, Attack Accuracy, Attacker Advantage over 50% chance baseline. | Non-private RF: AUC = 0.4998, Adv = 0.0000; DP-LR ($\epsilon=10$): AUC = 0.4992; DP-LR ($\epsilon=1.0$): AUC = 0.4897; DP-LR ($\epsilon=0.1$): AUC = 0.4977. | Attack AUC $\approx 0.50$ on regularized ensemble reflects low memorization for this architecture, but does not constitute a formal mathematical worst-case guarantee. |
| **Module 2: Differential Privacy for ML** | $(\epsilon, \delta)$-DP, pure $\epsilon$-DP, global sensitivity $\Delta f$, $L_2$ clipping bound $C$, objective perturbation, DP-SGD, Rényi Differential Privacy (RDP) accounting. | Regulatory compliance (GDPR Art. 25, CCPA) requires provable mathematical upper bounds on information leakage independent of adversary background knowledge. | **Dual Implementation:**<br>1. Pure $\epsilon$-DP Logistic Regression via IBM `diffprivlib` ($C=5.0$) across $\epsilon \in [0.1, 10.0]$.<br>2. Deep DP-SGD with PyTorch and Opacus featuring per-sample gradient clipping ($C=1.0$) and RDP accounting. | [`src/differential_privacy.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/differential_privacy.py) | Precision, Recall, F1-Score, PR-AUC, Achieved Privacy Budget $(\epsilon, \delta)$. | **diffprivlib Sweep:**<br>$\epsilon=0.1$: PR-AUC = 0.0968, F1 = 0.0815<br>$\epsilon=1.0$: PR-AUC = 0.1774, F1 = 0.1354<br>$\epsilon=10.0$: PR-AUC = 0.3275, F1 = 0.1214<br>**Opacus DP-SGD:**<br>Achieved $\epsilon = 4.9910, \delta = 10^{-5}$<br>Precision = 0.8043, Recall = 0.7551, F1 = 0.7789, PR-AUC = 0.6392. | Severe utility loss under pure linear DP at $\epsilon \le 0.1$ due to extreme class imbalance (0.17% fraud); deep DP-SGD with balanced loss significantly mitigates utility penalty. |
| **Module 3: Local Differential Privacy for ML** | Untrusted aggregator threat model, client-side perturbation, 1D bounded Laplace mechanism, Wasserstein distance, noise accumulation ($O(N/\epsilon_{loc}^2)$). | Cardholders or point-of-sale terminals randomize sensitive continuous attributes (e.g. transaction `Amount`) locally before transmitting to an untrusted central processor. | **Implemented Demonstration:** Client-side bounded Laplace perturbation on transaction `Amount` across $\epsilon_{loc} \in \{0.5, 1.0, 5.0\}$; statistical distortion analysis and downstream fraud classification evaluation. | [`src/local_differential_privacy.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/local_differential_privacy.py) | Mean Absolute Error (MAE), Mean Squared Error (MSE), Wasserstein Distance, Downstream PR-AUC and F1. | $\epsilon_{loc}=0.5$: WD = 4.8782, F1 = 0.8652, PR-AUC = 0.8595<br>$\epsilon_{loc}=1.0$: WD = 3.9346, F1 = 0.8603, PR-AUC = 0.8596<br>$\epsilon_{loc}=5.0$: WD = 1.2355, F1 = 0.8701, PR-AUC = 0.8588. | Perturbing only a single continuous feature (`Amount`) preserves downstream fraud detection because PCA features V1–V28 carry strong residual fraud signals. Perturbing all features locally incurs severe utility degradation. |
| **Module 4: Privacy-Preserving Data Mining (PPDM)** | Quasi-identifiers, continuous microaggregation, $k$-anonymity, Information Loss (SSE/SST), PCA vs. DP formal distinction. | Historical data anonymization for merchant sharing and cross-institutional fraud research without exposing unique individual transaction records. | **Empirical Demonstration & Analytical Critique:** Continuous microaggregation for $k \in \{5, 10\}$ using `MiniBatchKMeans`; Information Loss computation; formal proof of why PCA does NOT guarantee differential privacy. | [`src/privacy_preserving_mining.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/privacy_preserving_mining.py) | Normalized Information Loss (SSE/SST %), Precision, Recall, F1-Score, PR-AUC. | Raw Reference ($k=1$): Loss = 0.0%, F1 = 0.6000<br>$k=5$: Loss = 12.15%, F1 = 0.6667<br>$k=10$: Loss = 16.10%, F1 = 0.6000. | Averaging continuous feature vectors across clusters blurs fine-grained minority fraud signatures. PCA provides linear variance reduction, not provable privacy guarantees. |
| **Module 5: Compressive Privacy for ML** | Shannon Mutual Information, Information Bottleneck principle $\min [I(Z; S) - \beta I(Z; Y)]$, representation compression, adversarial min-max representation learning. | Preserving fraud detection utility while suppressing sensitive proxy attributes (e.g. cardholder wealth/tier profile) in representations shared with third-party vendors. | **Implemented Neural Architecture:** PyTorch Adversarial Compressive Autoencoder optimizing min-max objective across latent dimensions $d \in \{4, 8, 16\}$ with adversarial penalty $\lambda \in \{0.0, 0.5\}$ against sensitive amount tier proxy $S$. | [`src/compressive_privacy.py`](file:///C:/Users/Shreya/.gemini/antigravity/scratch/IA1_Fraud_Detection_Privacy/src/compressive_privacy.py) | Task Fraud F1-Score, Task PR-AUC, Adversary Sensitive Attribute Leakage AUC($S$). | **Latent Dim $d=4$:**<br>$\lambda=0.0$: Task F1 = 0.1015, Adv AUC($S$) = 0.7411<br>$\lambda=0.5$: Task F1 = 0.1273, Adv AUC($S$) = 0.5792 (Leakage suppressed by 16.2 points!)<br>**Latent Dim $d=8$:**<br>$\lambda=0.0$: Task F1 = 0.1916, Adv AUC($S$) = 0.8344<br>$\lambda=0.5$: Task F1 = 0.1076, Adv AUC($S$) = 0.5583 (Leakage suppressed by 27.6 points!). | Proxy attribute $S$ is derived from transaction volume; real-world cardholder demographic profiles (income, credit score) are not present in the public anonymized dataset. |

---

## 2. Assessment Rubric Breakdown (20 Marks Total)

### Stage 1: Problem Refinement and Objectives (4 Marks)
- Refines problem formulation to address the IA 1 zero-recall limitation caused by default 0.50 threshold.
- Formulates validation-only threshold tuning ($\tau^* = 0.39$) to achieve 83.67% test recall on the real dataset.
- Integrates multi-seed evaluation ($[42, 123, 456]$) for uncertainty quantification.

### Stage 2: Integration of Course Concepts (4 Marks)
- Fully addresses all 5 syllabus modules with working code and empirical evidence.
- Demonstrates central DP (Opacus DP-SGD and diffprivlib), local DP (Laplace mechanism), PPDM (microaggregation), and compressive privacy (adversarial autoencoder).

### Stage 3: Dataset Finalisation and Individual Implementation (4 Marks)
- Executes full workflow on the genuine 284,807-record Kaggle benchmark.
- Documents exact SHA-256 file hash, ODbL license, train/val/test split counts (199,364 / 28,481 / 56,962), and zero data leakage.

### Stage 4: Evaluation, Comparison, and Critical Analysis (4 Marks)
- Evaluates models using PR-AUC, Recall, Precision, F1-score, Confusion Matrix, and ROC-AUC.
- Compares default vs. tuned thresholds, baseline vs. DP models under MIA, and privacy vs. utility trade-offs.
- Profiles execution runtime (283.2 seconds total) and memory usage.

### Stage 5: Final Technical Documentation and Research/IP Development (4 Marks)
- Complete 24-section technical report in Markdown and DOCX formats.
- Complete, runnable Jupyter Notebook and beginner-friendly viva preparation guide.
- Downloadable ZIP archive containing all verified code and documentation.
