# Final Dataset Provenance & Data Integrity Record (IA 2)

**Subject:** Security and Privacy for Machine Learning (Semester VII)  
**Assessment:** IA 2 — Implementation, Evaluation and Technical Outcomes  
**Student Name:** Shreya Jadhav | **Roll Number:** 16014223077 | **Batch:** B3  
**Programme:** Honours in Applied Cyber Security  

---

## 1. Dataset Identification and Provenance Verification

- **Dataset Title:** Credit Card Fraud Detection
- **Host Repository:** [Kaggle Dataset: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Primary Research Group:** Machine Learning Group (MLG), Université Libre de Bruxelles (ULB) & Worldline.
- **Key Authors / Researchers:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.
- **Foundational Publication:**  
  Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *"Calibrating Probability with Undersampling for Unbalanced Classification."* IEEE Symposium Series on Computational Intelligence (SSCI), pp. 159–166, 2015.
- **License:** [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/)
- **Access / Local Verification Date:** October 2026

---

## 2. Cryptographic File Integrity & Schema Verification

| Parameter | Measured Specification | Verification Status |
| :--- | :--- | :--- |
| **File Name** | `creditcard.csv` | Verified |
| **File Size on Disk** | 150,828,752 bytes (~143.84 MB) | Verified |
| **Cryptographic Hash (SHA-256)** | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` | Verified |
| **Total Record Count (Rows)** | 284,807 transactions | Verified |
| **Total Attribute Count (Columns)** | 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`) | Verified |
| **Missing / Null Values** | 0 missing values across all columns | Verified |
| **Duplicate Records** | 1,081 duplicate transactions (retained for natural distribution) | Verified |
| **Legitimate Transactions (`Class = 0`)** | 284,315 records (99.8273%) | Verified |
| **Fraudulent Transactions (`Class = 1`)** | 492 records (0.1727%) | Verified |
| **Class Imbalance Ratio** | ~578 legitimate transactions per 1 fraudulent transaction | Verified |

---

## 3. Data Partitioning & Anti-Leakage Protocol

The dataset is partitioned into three mutually disjoint subsets using a **Stratified Shuffle Split** protocol (`random_state = 42`), strictly preserving the 0.1727% minority fraud prevalence across all partitions:

| Partition | Proportion | Total Records | Legitimate Transactions (`Class=0`) | Fraudulent Transactions (`Class=1`) | Fraud Rate (%) | Operational Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Split (`X_train`)** | 70.0% | 199,364 | 199,019 | 345 | 0.1730% | Model training, DP-SGD optimization, and scaler parameter estimation. |
| **Validation Split (`X_val`)** | 10.0% | 28,481 | 28,432 | 49 | 0.1720% | **Threshold tuning ($\tau^* = 0.39$)** and hyperparameter selection. |
| **Testing Split (`X_test`)** | 20.0% | 56,962 | 56,864 | 98 | 0.1720% | Final, unobserved evaluation of model utility and privacy attacks. |
| **Total Corpus** | 100.0% | 284,807 | 284,315 | 492 | 0.1727% | Full benchmark corpus. |

### Anti-Leakage Enforcement
1. **Zero Preprocessing Contamination:** The feature scaler (`RobustScaler`) is fitted **strictly on `X_train`**. Validation and testing sets are transformed using the pre-computed median and interquartile range (IQR) from the training set.
2. **Decision Boundary Independence:** In Experiment A, the classification decision threshold is tuned exclusively on `X_val`. The test set `X_test` remains completely unobserved until final metric evaluation.
3. **Disjoint Membership Inference Partitions:** In Experiment B, the dataset is split into four mutually exclusive partitions (Target Train, Target Holdout, Shadow Train, Shadow Holdout) to prevent data leakage between shadow attack training and target attack evaluation.

---

## 4. Responsible Data Use and Ethical Privacy Safeguards

1. **Feature Anonymization:** Attributes `V1` through `V28` are standardized principal components resulting from Principal Component Analysis (PCA) applied upstream by the dataset providers. Original financial identifiers (such as cardholder names, account numbers, merchant categories, and geolocation coordinates) were redacted prior to public release.
2. **Identity Non-Recovery Commitment:** In accordance with cybersecurity research ethics, no attempts were made to de-anonymize cardholders or perform linkage attacks with external auxiliary datasets.
3. **Distribution Compliance:** To comply with academic file-size submission constraints and open database redistribution ethics, the raw 150MB `creditcard.csv` file is excluded from the downloadable ZIP package. The repository provides automated lawful acquisition instructions and an offline synthetic verification generator (`data/synthetic_creditcard.csv`).
