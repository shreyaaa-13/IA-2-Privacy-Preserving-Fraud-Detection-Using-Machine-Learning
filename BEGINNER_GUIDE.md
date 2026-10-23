# Beginner's Guide & Viva Preparation Handbook

**Project:** Privacy-Preserving Fraud Detection Using Machine Learning  
**Subject:** Security and Privacy for Machine Learning (Semester VII)  
**Assessment:** IA 1 — Problem Formulation and Preliminary Implementation  

---

## 1. Project Directory & File Walkthrough

Here is a simple explanation of what every file in your project folder does:

```
IA1_Fraud_Detection_Privacy/
│
├── configs/
│   └── config.py               # Central settings: random seed (42), file paths, hyperparameters.
│
├── data/
│   ├── README.md               # Dataset explanation, Kaggle link, and lawful download options.
│   └── synthetic_creditcard.csv # Auto-generated test dataset (5,000 rows) for offline testing.
│
├── notebooks/
│   └── IA1_Fraud_Detection.ipynb # Interactive Jupyter Notebook for Google Colab or local VS Code.
│
├── src/
│   ├── __init__.py             # Makes 'src' a proper Python package.
│   ├── synthetic_data.py       # Generates synthetic data with ULB schema for unit testing.
│   ├── data_preprocessing.py   # Loads data, runs diagnostics, splits train/val/test, scales Amount.
│   ├── baseline_model.py       # Experiment A: Non-private Random Forest fraud classifier.
│   ├── membership_inference.py # Experiment B (Module 1): Controlled Membership Inference Attack.
│   ├── differential_privacy.py # Experiment C (Module 2): IBM diffprivlib DP Logistic Regression sweep.
│   ├── local_differential_privacy.py # Module 3: Client-side Laplace perturbation on Amount.
│   ├── privacy_preserving_mining.py  # Module 4: Continuous microaggregation & PCA vs DP distinction.
│   ├── compressive_privacy.py  # Module 5: Information Bottleneck representation compression.
│   └── evaluation.py           # Computes PR-AUC, F1, Recall, Confusion Matrix, and saves plots.
│
├── tests/
│   ├── test_preprocessing.py   # Unit tests for data loading, splitting, and leak-free scaling.
│   ├── test_metrics.py         # Unit tests for precision, recall, F1, and edge cases.
│   └── test_pipeline.py        # Integration test running the entire pipeline on mock data.
│
├── results/                    # Generated JSON metrics and consolidated comparison CSV tables.
├── figures/                    # Generated high-resolution diagnostic plots (PR curve, ROC, CM).
├── reports/
│   ├── IA1_Preliminary_Report.md # Formal 18-section academic technical report ready for submission.
│   ├── module_mapping.md       # Syllabus mapping table and marks rubric breakdown.
│   └── dataset_provenance.md   # Official data source, licensing (ODbL), and ethical usage rules.
│
├── run_experiments.py          # Master terminal script to run all experiments with one command.
├── requirements.txt            # List of required Python packages with compatible versions.
├── .gitignore                  # Prevents committing large datasets, virtualenvs, and cache files.
└── README.md                   # Top-level project landing page and documentation overview.
```

---

## 2. How to Run the Project (Step-by-Step)

### Option A: Running in Google Colab (Recommended for College Viva / Beginners)
1. Open Google Chrome and visit [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `notebooks/IA1_Fraud_Detection.ipynb`.
3. In the Colab file explorer (left sidebar folder icon), upload the `src/` and `configs/` folders, or clone your project repository.
4. Run the first cell to automatically install `diffprivlib`.
5. Click **Runtime → Run all**.
6. The notebook will automatically run using the synthetic verification harness if `creditcard.csv` is not present, producing all charts and tables directly inline!

### Option B: Running Locally on Windows in VS Code
1. Open the project folder `IA1_Fraud_Detection_Privacy` in Visual Studio Code.
2. Open a new PowerShell terminal (`Ctrl + ~`).
3. (Optional) Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Run the automated unit tests to verify your environment:
   ```powershell
   pytest tests/
   ```
   *(Expected output: `13 passed in ~4.5s`)*
6. Run the master pipeline in test mode (fast, offline):
   ```powershell
   python run_experiments.py --mode test
   ```
7. To run on the real 284,807-record Kaggle dataset:
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
   - Place `creditcard.csv` inside the `data/` folder.
   - Run:
     ```powershell
     python run_experiments.py --mode real
     ```

---

## 3. Where are Outputs Saved?

- **Metrics Tables:**
  - `results/experiments_comparison.csv`: Clean CSV table with all models, precision, recall, F1, PR-AUC, and privacy parameters.
  - `results/all_experiments_metrics.json`: Detailed machine-readable JSON containing full metrics.
- **Visual Plots:**
  - `figures/baseline_confusion_matrix_stratified.png`: Confusion matrix heatmap.
  - `figures/baseline_pr_curve_stratified.png`: Precision-Recall curve with baseline comparison.
  - `figures/baseline_roc_curve_stratified.png`: ROC curve with AUC score.
  - `figures/mia_roc_curve.png`: Attacker ROC curve for Membership Inference.
  - `figures/dp_privacy_utility_tradeoff.png`: Privacy budget $\epsilon$ vs. F1-score & PR-AUC.
  - `figures/ldp_feature_distributions.png`: Histograms of transaction `Amount` under varying local noise.

---

## 4. How to Interpret Your Results

1. **Why is Accuracy ~99.8% but Recall or F1 can be 0.0?**
   - In fraud detection, legitimate transactions make up 99.83% of the dataset. A "dumb" model that always says "legitimate" gets 99.83% accuracy while catching 0 frauds.
   - Always look at **Recall** (how many frauds did we catch?) and **PR-AUC** (how reliable is the model across all fraud thresholds?).
2. **How do I interpret the Membership Inference Attack (MIA) AUC?**
   - The attack evaluation uses an equal 50/50 mix of training members and unseen holdout records.
   - A random guess has an **AUC of 0.50**.
   - If Attack AUC is **0.50 to 0.55**, the model has **low privacy leakage** (it is not memorizing training records).
   - If Attack AUC exceeds **0.70**, the model has **severe privacy leakage** (it has memorized training transactions).
3. **How do I interpret the Privacy Budget $\epsilon$ (Epsilon)?**
   - **Smaller $\epsilon$ (e.g., 0.1):** High noise, very strong privacy protection, lower utility (F1/PR-AUC drops).
   - **Larger $\epsilon$ (e.g., 5.0 or 10.0):** Low noise, weaker privacy protection, high utility (closer to non-private baseline).
   - This relationship is known as the **Privacy-Utility Trade-Off**.

---

## 5. Viva Preparation: 12 Likely Questions & Simple Model Answers

### Q1: Why did you choose Random Forest as your baseline model?
**Answer:**  
"Random Forest is an ensemble of decision trees that handles non-linear feature interactions well and is a strong, widely accepted baseline in banking fraud detection. Furthermore, its bagging (bootstrap aggregation) mechanism and maximum depth constraints provide some natural resistance against overfitting, which helps reduce membership leakage compared to a single unpruned decision tree."

### Q2: Why is accuracy misleading in this project?
**Answer:**  
"Because of extreme class imbalance: out of 284,807 transactions, only 492 are fraudulent (~0.17%). A trivial classifier predicting class 0 for every transaction achieves 99.83% accuracy but 0.0% recall, detecting no fraud whatsoever. Therefore, Precision-Recall AUC (PR-AUC), Sensitivity (Recall), and F1-score are the authoritative metrics."

### Q3: What is a Membership Inference Attack (MIA) and what is the attacker's goal?
**Answer:**  
"The target model's goal is fraud classification: given a transaction $x$, predict fraud label $y \in \{0, 1\}$.  
The attacker's goal in MIA is privacy violation: given a known transaction $(x, y)$, predict membership $m \in \{0, 1\}$ (whether this transaction was part of the model's private training dataset). The attacker exploits the fact that machine learning models usually exhibit higher confidence and lower prediction entropy on data they were trained on compared to unseen test data."

### Q4: How did you ensure your Membership Inference experiment was leak-free?
**Answer:**  
"We partitioned the data into four strictly disjoint subsets: Target Train, Target Holdout, Shadow Train, and Shadow Holdout. The shadow model was trained on the Shadow Train set to simulate the target model, and the attack classifier was trained on the shadow model's outputs. The attack classifier was then evaluated on the Target data, which the attacker had never observed during training. The evaluation set was balanced 50/50 between members and non-members, meaning random guessing yields exactly 50% accuracy."

### Q5: What is Differential Privacy ($\epsilon$-DP) in simple terms?
**Answer:**  
"Differential privacy is a mathematical guarantee that the output of an algorithm does not depend significantly on whether any single individual's data was included in the training set or not. The parameter $\epsilon$ (epsilon) is our privacy budget: smaller $\epsilon$ means stronger privacy and more noise, while larger $\epsilon$ means less noise and higher model accuracy."

### Q6: Why did you use IBM `diffprivlib` instead of just adding random Gaussian noise to the data?
**Answer:**  
"Adding arbitrary Gaussian noise to input features is heuristic and does NOT satisfy formal differential privacy. Formal differential privacy requires bounding the global sensitivity $\Delta f$ (via $L_2$ feature clipping $C$) and adding noise calibrated specifically to the query sensitivity divided by epsilon ($\Delta f / \epsilon$). IBM `diffprivlib` implements provable objective perturbation for logistic regression with formal theoretical privacy guarantees."

### Q7: What is the difference between Central Differential Privacy and Local Differential Privacy?
**Answer:**  
"In Central DP, users send their raw data to a trusted central server, and the server adds noise before releasing the model. In Local DP, there is NO trusted server: each user randomizes their data on their own local device before transmitting it. Because noise is added to every record individually in Local DP, the accumulated noise is much larger, causing a steeper utility drop."

### Q8: Does Principal Component Analysis (PCA) guarantee Differential Privacy?
**Answer:**  
"No, absolutely not. PCA is simply an orthogonal linear dimensionality reduction technique designed to maximize variance ($Z = X W$). It contains no calibrated noise, no sensitivity bounds, and preserves high-fidelity sample reconstruction ($X \approx Z W^T$). In the Kaggle ULB dataset, PCA was used upstream by the bank for basic feature redaction, not as a mathematical privacy guarantee."

### Q9: What is Continuous Microaggregation in PPDM?
**Answer:**  
"Microaggregation is a data anonymization technique for continuous numerical features. It groups similar records into clusters of size at least $k$ using distance metrics (like k-means) and replaces each individual record with the cluster centroid. This ensures that every record is identical to at least $k-1$ other records, achieving $k$-anonymity. However, in fraud detection, this averaging can blur the unique signatures of rare fraud cases, causing a sharp drop in recall."

### Q10: What is Compressive Privacy and the Information Bottleneck principle?
**Answer:**  
"Compressive Privacy formalizes privacy using Shannon Mutual Information. The objective is:
$$\min [I(Z; S) - \beta I(Z; Y)] \quad \text{subject to } I(Z; X) \le R$$
It compresses the raw input $X$ into a latent representation $Z$ such that mutual information with a sensitive attribute $S$ is minimized (privacy), while mutual information with the utility task $Y$ (fraud detection) is maximized."

### Q11: How did you prevent data leakage during preprocessing?
**Answer:**  
"We applied train-test splitting before fitting any feature transformation. Specifically, our `RobustScaler` was fitted strictly on the training set features. The validation and test sets were transformed using the pre-computed median and IQR of the training set. No test distribution statistics were ever exposed during preprocessing."

### Q12: What work remains to be completed for IA 2?
**Answer:**  
"For IA 2, we will:
1. Benchmark full training runtimes and memory consumption on the entire 284,807-record Kaggle dataset;
2. Implement Deep Learning DP-SGD with Rényi Differential Privacy (RDP) accounting using PyTorch and Opacus;
3. Build an adversarial neural autoencoder to optimize the Compressive Privacy min-max Information Bottleneck;
4. Tune probability decision thresholds to maximize recall at fixed precision under differential privacy."

---

## 6. What You Should Be Able to Explain Independently in Your Viva

During your viva, focus on explaining these 4 core concepts clearly:
1. **The Core Trade-off:** Explain why fraud detection is hard (0.17% class imbalance) and why privacy makes it harder (noise masks rare fraud events).
2. **The Attacker's Strategy:** Explain how an attacker uses shadow models to perform Membership Inference by observing prediction confidence.
3. **The Meaning of Epsilon:** Explain that smaller $\epsilon$ means higher privacy and lower model accuracy, pointing to your Privacy-Utility plot (`figures/dp_privacy_utility_tradeoff.png`).
4. **The PCA Myth:** Clearly state that PCA is NOT differential privacy because it adds zero noise and has no mathematical privacy guarantees.
