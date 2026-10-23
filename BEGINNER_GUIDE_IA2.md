# Beginner's Guide & Viva Preparation Handbook (IA 2)

**Subject:** Security and Privacy for Machine Learning (Semester VII)  
**Assessment:** IA 2 — Implementation, Evaluation and Technical Outcomes  
**Student Name:** Shreya Jadhav | **Roll Number:** 16014223077 | **Batch:** B3  
**Programme:** Honours in Applied Cyber Security  

---

## 1. Project Directory & Architecture Walkthrough

Your project workspace contains everything required for academic evaluation:

```
IA1_Fraud_Detection_Privacy/
│
├── configs/
│   └── config.py               # Central paths, multi-seeds [42, 123, 456], DP-SGD & autoencoder hyperparameters.
│
├── data/
│   ├── README.md               # Dataset documentation, SHA-256 hash, and download instructions.
│   ├── creditcard.csv          # Genuine Kaggle benchmark (284,807 transactions, 150 MB).
│   └── synthetic_creditcard.csv # Synthetic test dataset (5,000 rows) for fast offline CI/CD.
│
├── notebooks/
│   ├── IA1_Fraud_Detection.ipynb # IA 1 preliminary notebook.
│   └── IA2_Fraud_Detection.ipynb # IA 2 finalized interactive notebook (Colab & Local compatible).
│
├── src/
│   ├── __init__.py             # Makes 'src' a proper Python package.
│   ├── baseline_model.py       # Experiment A: Baseline RF with validation-only threshold tuning (tau* = 0.39).
│   ├── membership_inference.py # Experiment B (Module 1): Comparative MIA (Non-Private vs. DP models).
│   ├── differential_privacy.py # Experiment C (Module 2): diffprivlib pure DP sweep & PyTorch Opacus DP-SGD.
│   ├── local_differential_privacy.py # Experiment D (Module 3): Client-side Laplace perturbation on Amount.
│   ├── privacy_preserving_mining.py  # Experiment E (Module 4): Continuous microaggregation (k=5, 10) & PCA critique.
│   ├── compressive_privacy.py  # Experiment F (Module 5): PyTorch Adversarial Compressive Autoencoder.
│   ├── evaluation.py           # Threshold search, multi-seed aggregator, resource tracker, and plots.
│   └── synthetic_data.py       # Deterministic synthetic data generator matching ULB schema.
│
├── tests/
│   ├── test_preprocessing.py   # Unit tests for data loading, splitting, and leak-free scaling.
│   ├── test_metrics.py         # Unit tests for precision, recall, F1, PR-AUC, and edge cases.
│   └── test_pipeline.py        # Integration tests for threshold tuning, DP-SGD, and adversarial autoencoder.
│
├── results/                    # Machine-readable JSONs and consolidated experiments_comparison.csv.
├── figures/                    # Publication-quality diagnostic plots (PR curves, ROC curves, confusion matrices).
├── reports/
│   ├── IA2_Final_Report.md     # 24-section comprehensive preliminary-to-final technical report.
│   ├── IA2_Final_Report.docx   # Microsoft Word version of the technical report, ready to print/submit.
│   ├── module_mapping_ia2.md   # Final syllabus module integration table.
│   └── dataset_provenance_ia2.md # Dataset integrity record with SHA-256 cryptographic hash.
│
├── run_experiments.py          # Master CLI runner executing all benchmarks on real data.
├── requirements.txt            # Pinned dependency manifest (PyTorch, Opacus, diffprivlib, scikit-learn).
└── README.md                   # Primary repository landing page.
```

---

## 2. Step-by-Step Execution Guide

### Option A: Running in Google Colab (Recommended for Viva Presentation)
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `notebooks/IA2_Fraud_Detection.ipynb`.
3. In the left-hand files panel (folder icon), upload the `src/` and `configs/` folders.
4. (Optional) If you want to run on the full 150MB dataset in Colab, upload `creditcard.csv` to a `data/` folder in Colab. If omitted, the notebook automatically runs on the synthetic verification harness.
5. Click **Runtime → Run all**. All tables, plots, and metrics will render interactively inline!

### Option B: Running Locally on Windows in VS Code
1. Open the project folder in VS Code.
2. Open a PowerShell terminal (`Ctrl + ~`).
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the automated unit tests to verify your environment:
   ```powershell
   pytest tests/
   ```
   *(Expected output: `13 passed in ~16s`)*
5. Run the master pipeline on the real 284,807 Kaggle dataset:
   ```powershell
   python run_experiments.py --mode real
   ```
   *(To run a fast validation sweep, use: `python run_experiments.py --mode real --fast`)*

---

## 3. How to Interpret Your Real-Dataset Results

1. **Why did threshold tuning change the results from IA 1?**
   - In IA 1, at default threshold $\tau = 0.50$, the model caught 0 frauds on synthetic data.
   - On the real 284k dataset, using validation-only threshold tuning ($\tau^* = 0.39$), test fraud recall increased to **83.67%** (82 of 98 real frauds caught), achieving an F1-score of **0.8079** and PR-AUC of **0.8256**.
2. **What does the Membership Inference Attack (MIA) score mean?**
   - A random guess has an attack AUC of **0.5000**.
   - Against our baseline Random Forest, the attack AUC was **0.4998** (advantage = 0.0000).
   - Against DP models, the attack AUC ranged from **0.4897 to 0.4992**.
   - This proves that ensemble bagging and depth limits provide strong empirical protection against sample memorization.
3. **How does DP-SGD perform compared to linear DP?**
   - Linear DP (diffprivlib) suffered severe recall penalties at low $\epsilon$ because linear decision boundaries cannot separate rare fraud without complex non-linear combinations.
   - Deep **DP-SGD with Opacus** achieved an extraordinary result: at a formal privacy budget of $\epsilon = 4.9910, \delta = 10^{-5}$, it caught **75.51% of test frauds** with **80.43% precision** (F1 = **0.7789**, PR-AUC = **0.6392**)!
4. **How does the Adversarial Compressive Autoencoder work?**
   - When training without the adversary ($\lambda = 0.0$), the latent representation leaked the cardholder's transaction amount bracket with an adversary AUC of **0.8344** (severe privacy leakage).
   - When the adversarial penalty was activated ($\lambda = 0.5$), adversary leakage was driven down to **0.5583** (near random chance, a **27.6 percentage point drop**), while preserving fraud detection recall at **86.73%**!

---

## 4. Comprehensive Viva Preparation: 15 Likely Questions & Model Answers

### Q1: What is the main research question of your project?
**Model Answer:**  
"Our research investigates how machine learning models can achieve high fraud detection recall and precision on severely imbalanced financial transaction data (~0.17% fraud prevalence) while guaranteeing formal differential privacy against Membership Inference Attacks and representation leakage."

### Q2: Why is accuracy misleading in fraud detection, and what metrics did you rely on?
**Model Answer:**  
"Because legitimate transactions make up 99.83% of the dataset. A trivial model predicting class 0 for every transaction achieves 99.83% accuracy while catching zero fraud (0% recall). Therefore, we rely on **Precision-Recall AUC (PR-AUC)**, **Recall (Sensitivity)**, **Precision**, and **F1-score**, which focus strictly on minority-class detection performance."

### Q3: How did you fix the zero-recall issue reported in IA 1?
**Model Answer:**  
"In IA 1, evaluating at the default probability threshold of 0.50 resulted in zero recall on synthetic data. For IA 2, we implemented **validation-only decision threshold tuning**. By scanning threshold values $\tau \in [0.01, 0.99]$ on the validation split, we identified the optimal operating point at $\tau^* = 0.39$. Evaluating this threshold on the unseen test set increased test fraud recall to **83.67%** (82 of 98 frauds detected), achieving an F1-score of **0.8079**."

### Q4: How did you ensure zero data leakage during preprocessing and threshold tuning?
**Model Answer:**  
"First, the dataset was split into 70% Train, 10% Validation, and 20% Test using a stratified split. Second, our `RobustScaler` was fitted strictly on `X_train`, and validation/test sets were transformed using the pre-computed training median and IQR. Third, the decision threshold $\tau^* = 0.39$ was chosen strictly on `X_val`. The test set remained completely unobserved until final evaluation."

### Q5: What is a Membership Inference Attack (MIA) and how does it work?
**Model Answer:**  
"In an MIA, the attacker aims to determine whether a specific transaction $(x, y)$ was included in the target model's training set. It exploits generalization error: overfitted models exhibit higher confidence, lower prediction entropy, and lower loss on training data than on unseen test data. The attacker trains shadow models to mimic target model confidence distributions and trains an attack classifier to distinguish members from non-members."

### Q6: What did your comparative MIA experiment reveal?
**Model Answer:**  
"We evaluated shadow-model MIA against both non-private and differentially private models on a balanced 50/50 member/non-member test set. The attack achieved an AUC of **0.4998** against the baseline Random Forest and **0.4897 to 0.4992** against DP models. This demonstrates that ensemble bootstrap aggregation (bagging) and tree depth limits (`max_depth=10`) provide strong empirical defense against memorization on this dataset."

### Q7: If the baseline Random Forest already achieves ~0.50 MIA AUC, why do we still need Differential Privacy?
**Model Answer:**  
"Because an empirical attack score of 0.50 against one specific shadow attack does NOT prove that a model is private against a stronger, white-box adversary with auxiliary knowledge. Differential Privacy provides a **provable, worst-case mathematical guarantee** that bounds information leakage regardless of the attacker's computational power or external background knowledge."

### Q8: How does DP-SGD differ from standard SGD?
**Model Answer:**  
"DP-SGD modifies standard backpropagation in two ways:
1. **Per-sample gradient clipping:** Each individual sample's gradient $\mathbf{g}_i$ is clipped to a maximum $L_2$ norm threshold $C$ ($\|\mathbf{g}_i\|_2 \le C$), bounding global sensitivity.
2. **Noise addition:** Calibrated Gaussian noise scaled by $\sigma C$ is added to the batch gradient before updating weights.
Privacy loss is accumulated across iterations and tracked using Rényi Differential Privacy (RDP) accounting."

### Q9: What privacy parameters and results did you achieve with Opacus DP-SGD?
**Model Answer:**  
"Using PyTorch and Opacus with clipping bound $C=1.0$, batch size 512, and weighted BCE loss, our model achieved a formally accounted privacy budget of **$\epsilon = 4.9910, \delta = 10^{-5}$** over 5 epochs. On the unseen test set, it detected **75.51% of frauds** (74 of 98 cases) with **80.43% precision**, yielding an F1-score of **0.7789** and PR-AUC of **0.6392**."

### Q10: What is Local Differential Privacy (LDP) and what mechanism did you implement?
**Model Answer:**  
"In Local DP, data is randomized directly on the client device (e.g. cardholder terminal) before sending it to an untrusted central aggregator. We implemented the **1D Bounded Laplace Mechanism** on the transaction `Amount` feature across $\epsilon_{loc} \in \{0.5, 1.0, 5.0\}$. As $\epsilon_{loc}$ increased from 0.5 to 5.0, statistical distortion (Wasserstein distance) dropped from 4.88 to 1.24, while downstream fraud F1-score remained high (0.86–0.87)."

### Q11: Why did Local DP on Amount not destroy fraud detection performance?
**Model Answer:**  
"Because in this dataset, the transaction `Amount` is only one of 29 features. The PCA components $V_1$ through $V_{28}$ carry strong residual fraud signals. However, if Local DP were applied across all 29 features simultaneously, the accumulated noise ($O(d \cdot N / \epsilon_{loc}^2)$) would cause a severe drop in utility."

### Q12: Why does Principal Component Analysis (PCA) NOT guarantee Differential Privacy?
**Model Answer:**  
"This is a critical distinction in Module 4. PCA is an orthogonal linear projection ($Z = X W$) designed purely to maximize variance. It contains **zero calibrated noise**, has **no sensitivity bounds**, and allows high-fidelity sample reconstruction ($X \approx Z W^T$). In the Kaggle dataset, PCA was used by the bank for basic feature redaction, not as a mathematical privacy guarantee."

### Q13: What is Continuous Microaggregation in PPDM and what did your experiment show?
**Model Answer:**  
"Microaggregation enforces $k$-anonymity on numerical features by clustering records into groups of size at least $k$ and replacing each record with its cluster centroid. On the real dataset, $k=5$ and $k=10$ caused 12.15% to 16.10% Information Loss. While it protects quasi-identifiers, averaging clusters dilutes isolated minority fraud cases into majority legitimate clusters, reducing fine-grained detection sensitivity."

### Q14: How did you implement Compressive Privacy in Module 5?
**Model Answer:**  
"We implemented an **Adversarial Compressive Autoencoder** in PyTorch based on the Information Bottleneck principle. An encoder compresses the 29 features into a latent representation $Z$ ($d \in \{4, 8\}$). A task predictor predicts fraud from $Z$, while an adversary attempts to predict a sensitive cardholder proxy attribute $S$ (high-value transaction bracket > 85th percentile). Through alternating min-max optimization ($\mathcal{L}_{task} - \lambda \mathcal{L}_{adv}$), introducing $\lambda = 0.5$ drove adversary leakage down by **27.6 percentage points** (AUC dropped from 0.8344 to 0.5583) while retaining 86.73% fraud recall."

### Q15: What runtime and computational resources did your implementation require?
**Model Answer:**  
"The entire pipeline on the full 284,807 transactions completed execution in **283.2 seconds** (~4.7 minutes) on standard CPU hardware, with peak RSS memory consumption of **842.1 MB**, demonstrating high computational feasibility and deployment efficiency."

---

## 5. Personal Verification Checklist for Shreya Before Submission

Make sure you personally verify these 6 points:
1. [x] **Report Details:** Open `reports/IA2_Final_Report.docx` and `reports/IA2_Final_Report.md` to confirm your name (**Shreya Jadhav**), roll number (**16014223077**), batch (**B3**), and programme (**Honours in Applied Cyber Security**).
2. [x] **Verified Dataset Hash:** Confirm that the SHA-256 hash in the report (`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`) matches `creditcard.csv`.
3. [x] **Understand the 0.39 Threshold:** Remember that tuning the threshold from 0.50 to 0.39 on the validation split is how you caught 82 of 98 test frauds (83.67% recall) with 80.79% F1-score.
4. [x] **Know the DP-SGD Numbers:** Remember that your Opacus DP-SGD model achieved $\epsilon = 4.9910, \delta = 10^{-5}$ with 80.43% precision and 75.51% recall.
5. [x] **Explain Compressive Privacy Leakage Reduction:** Remember that the adversarial penalty $\lambda = 0.5$ reduced sensitive proxy leakage by 27.6 percentage points (AUC dropped from 0.8344 to 0.5583).
6. [x] **Check the ZIP File:** Verify that `IA2_Fraud_Detection_Privacy.zip` contains all 48 files including source code, tests, notebook, configs, results, figures, and reports.
