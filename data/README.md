# Dataset Directory & Download Instructions

This folder holds the transaction data for the **Privacy-Preserving Fraud Detection** project.

---

## 1. Important Academic & Licensing Notice

The genuine benchmark dataset is the **ULB Credit Card Fraud Detection** dataset hosted on Kaggle:
`https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`

- **Record Count:** 284,807 transactions (492 frauds, 0.1727% fraud prevalence).
- **License:** Open Database License (ODbL) v1.0.
- **File Size:** ~150 MB uncompressed (`creditcard.csv`).

To comply with licensing distribution conditions and maintain a lightweight submission archive, the raw 150MB `creditcard.csv` file is **excluded from the ZIP archive**.

---

## 2. How to Obtain `creditcard.csv`

### Manual Download (Recommended)
1. Visit the Kaggle dataset page:  
   [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click the **Download** button (requires a free Kaggle account).
3. Unzip the downloaded archive to locate `creditcard.csv`.
4. Copy `creditcard.csv` directly into this `data/` directory:
   `IA1_Fraud_Detection_Privacy/data/creditcard.csv`

### Kaggle CLI Download
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p data/
```

---

## 3. Offline Verification Dataset

To allow immediate, offline software verification without waiting to download 150MB, the project includes an automated synthetic data generator:
- `data/synthetic_creditcard.csv` (5,000 transactions, 25 fraud records, matching the exact 31-column schema).
- When `creditcard.csv` is not present, the entire codebase automatically falls back to `synthetic_creditcard.csv`, allowing all tests and the notebook to execute out-of-the-box.
- When `creditcard.csv` is added, the pipeline automatically detects and uses it.
