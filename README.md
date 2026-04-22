# Synthesize or Reconstruct: GAN-Based Approaches to Credit Card Fraud Detection

Companion code for the paper *"Synthesize or Reconstruct: A Methodologically Corrected Benchmark of GAN-Based Approaches to Credit Card Fraud Detection."*

This repository provides a fully reproducible, leakage-free benchmark that systematically corrects four critical methodological flaws found in prior work, then evaluates six GAN variants across two detection paradigms on the ULB Credit Card Fraud dataset.

---

## Overview

Credit card fraud detection literature is rife with inflated results caused by data leakage. This work:

1. **Identifies and documents four critical flaws** in a widely-cited prior pipeline (validated against Hayat & Magnier, 2025).
2. **Implements a corrected baseline pipeline** that eliminates all identified leakage.
3. **Benchmarks six GAN architectures** split across two paradigms:
   - **Pipeline A — Synthesize**: GANs generate synthetic minority-class samples; a LightGBM classifier trains on augmented data.
   - **Pipeline B — Reconstruct**: GANs learn the distribution of normal transactions; anomalies are flagged by reconstruction error.

---

## Flaws Corrected

| # | Severity | Flaw | Fix |
|---|----------|------|-----|
| 1 | Critical | Train-holdout record overlap — all 492 fraud records appeared in both training and evaluation sets | Temporal holdout carved from the **end** of the time-sorted dataset before any model sees data; overlap verified by identity hash |
| 2 | Critical | SMOTE applied before K-fold construction — synthetic samples from fold-k training leaked into fold-k validation | SMOTE runs **inside** each fold's training partition, after fold construction |
| 3 | Critical | Threshold selection on the final test set — same 492-fraud holdout used for both threshold sweep and final metric reporting | Threshold selected on a dedicated `final_val` partition drawn from `train_pool`; holdout accessed exactly once |
| 4 | Serious | Artificial evaluation class distribution (~9% fraud) instead of the real-world rate (~0.17%) | Holdout retains the natural class distribution throughout |

---

## Models

### Pipeline A — Synthesize (Generative Augmentation → LightGBM)

| Notebook | Architecture | Training |
|----------|-------------|----------|
| [`notebooks/ctgan.ipynb`](notebooks/ctgan.ipynb) | **CTGAN** — Conditional Tabular GAN with mode-specific normalisation and PacGAN discriminator | Standard GAN loss |
| [`notebooks/tgan.ipynb`](notebooks/tgan.ipynb) | **T-GAN** — Transformer encoder generator + discriminator | WGAN-GP |
| [`notebooks/rgan.ipynb`](notebooks/rgan.ipynb) | **R-GAN** — Recurrent GAN (LSTM-based generator and discriminator) | WGAN-GP |

All Pipeline A models generate synthetic fraud samples → concatenate with real training data → train a LightGBM classifier → evaluate on the natural-distribution holdout.

### Pipeline B — Reconstruct (Anomaly Detection on Normal Transactions Only)

| Notebook | Architecture | Anomaly Score |
|----------|-------------|---------------|
| [`notebooks/ganomaly.ipynb`](notebooks/ganomaly.ipynb) | **GANomaly** — Encoder–Generator–Encoder with adversarial + reconstruction + latent-space losses | Latent-space discrepancy ‖z − ẑ‖₂ |
| [`notebooks/skip-ganomaly.ipynb`](notebooks/skip-ganomaly.ipynb) | **Skip-GANomaly** — U-Net skip connections between encoder and decoder | Latent-space discrepancy with skip-augmented reconstruction |
| [`notebooks/fanogan.ipynb`](notebooks/fanogan.ipynb) | **f-AnoGAN** — Fast Unsupervised Anomaly Detection with encoder trained post-WGAN | Combined feature-space residual score |

All Pipeline B models train **only on normal transactions**; fraud is never seen during training.

---

## Repository Structure

```
synthesize-or-reconstruct/
├── notebooks/
│   ├── pipeline.ipynb          # Corrected baseline pipeline (all fixes documented)
│   ├── ctgan.ipynb             # Pipeline A: CTGAN
│   ├── tgan.ipynb              # Pipeline A: Transformer GAN
│   ├── rgan.ipynb              # Pipeline A: Recurrent GAN
│   ├── ganomaly.ipynb          # Pipeline B: GANomaly
│   ├── skip-ganomaly.ipynb     # Pipeline B: Skip-GANomaly
│   ├── fanogan.ipynb           # Pipeline B: f-AnoGAN
│   └── creditcard.csv          # ULB dataset (tracked via Git LFS)
├── feature_store/
│   ├── pipeline_a_train.parquet   # SMOTE-ready training split for Pipeline A
│   ├── pipeline_a_val.parquet     # Validation split for Pipeline A
│   ├── pipeline_b_train.parquet   # Normal-only training split for Pipeline B
│   └── holdout.parquet            # Natural-distribution holdout (never used for tuning)
└── research/
    └── ...                        # Supporting research documents
```

---

## Dataset

**ULB Credit Card Fraud Detection** — European cardholders, September 2013.

- 284,807 transactions over two days; 492 frauds (0.172% prevalence).
- Features: `Time`, `V1`–`V28` (PCA-anonymised), `Amount`, `Class`.
- `Time` is excluded from model features (Fix 8, following Hayat & Magnier) to prevent temporal leakage.

> Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place as `notebooks/creditcard.csv`, or let the notebook fetch it via Git LFS.

---

## Quickstart

All notebooks are designed to run on **Google Colab** (free GPU tier is sufficient for all experiments). Open any notebook and run from top to bottom.

### Running on Colab

```python
# Each notebook self-downloads its feature-store parquets from this repo:
BASE = "https://raw.githubusercontent.com/kkipngenokoech/synthesize-or-reconstruct/main/feature_store/"
```

No local setup is required. The feature store is pre-computed and version-locked to ensure every notebook starts from identical data splits.

### Running Locally

```bash
# Clone with LFS to get creditcard.csv
git lfs install
git clone https://github.com/kkipngenokoech/synthesize-or-reconstruct.git
cd synthesize-or-reconstruct

pip install torch lightgbm scikit-learn imbalanced-learn ctgan wandb pandas numpy scipy
jupyter notebook notebooks/pipeline.ipynb
```

### Experiment Tracking

All notebooks log to [Weights & Biases](https://wandb.ai). Replace the API key in the `wandb.login()` cell with your own key from [wandb.ai/settings](https://wandb.ai/settings).

---

## Evaluation Protocol

All models are evaluated against the same temporal holdout that retains the natural class distribution (≈0.17% fraud). Reported metrics:

| Metric | Rationale |
|--------|-----------|
| AUPRC (Average Precision) | Primary metric — more informative than AUROC under extreme class imbalance |
| AUROC | Standard ranking metric |
| F1 Score | Harmonic mean at the operating threshold |
| G-Mean | Geometric mean of sensitivity and specificity |
| MCC | Matthews Correlation Coefficient — robust to class imbalance |
| Precision / Recall | At the selected operating threshold |

Thresholds for all models are selected on `pipeline_a_val` or a held-aside `final_val` partition — **never on the holdout**.

---

## Reproducibility

- Random seeds are fixed (`numpy`, `torch`, `random`) inside every notebook.
- Feature store parquets are deterministically generated by [`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb) and committed to the repo, so GAN notebooks do not depend on re-running the baseline.
- Schema validation (column count, fraud rate, null check) runs at data-load time in every notebook and raises on mismatch.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | T-GAN, R-GAN, GANomaly, Skip-GANomaly, f-AnoGAN |
| `ctgan` | CTGAN model |
| `lightgbm` | Downstream classifier (Pipeline A) |
| `scikit-learn` | Preprocessing, metrics, K-fold |
| `imbalanced-learn` | SMOTE, G-Mean metric |
| `wandb` | Experiment tracking |
| `pandas`, `numpy`, `scipy` | Data handling |

---

## Citation

If you use this code or the corrected pipeline in your work, please cite:

```bibtex
@article{koech2025synthesize,
  title   = {Synthesize or Reconstruct: A Methodologically Corrected Benchmark of
             GAN-Based Approaches to Credit Card Fraud Detection},
  author  = {Koech, Kipngeno},
  year    = {2025},
}
```

This work builds on the leakage taxonomy of:

```bibtex
@article{hayat2025data,
  title   = {Data Leakage and Deceptive Performance},
  author  = {Hayat, Youssef and Magnier, Baptiste},
  journal = {Mathematics},
  volume  = {13},
  number  = {16},
  pages   = {2563},
  year    = {2025},
  url     = {https://www.mdpi.com/2227-7390/13/16/2563}
}
```

---

## License

MIT
