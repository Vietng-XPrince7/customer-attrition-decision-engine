# Dual-Stage Revenue Defense & Profit-Driven Attrition Engine

> *"Preventing churn is not a retention problem — it is a revenue compression problem solved through rigorous economic decision theory."*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-green)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3.5+-purple)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Executive Overview

Banks lose between **15–25% of their retail customer base annually**, but raw churn rates obscure a more insidious threat: **Revenue Compression** — customers who remain enrolled but steadily reduce balances, downgrade products, and reduce transaction frequency. By the time they formally close their accounts, up to **60% of their lifetime value has already been lost**.

This project implements a **Dual-Stage ML system** that attacks both failure modes simultaneously:

| Stage | Problem | Method | Business Outcome |
|---|---|---|---|
| **Stage 1** | Revenue Compression Detection | LightGBM Regressor/Classifier | Flag balance-shrinking customers 90 days early |
| **Stage 2** | Churn Probability Calibration | Stacking Ensemble + Isotonic Calibration | Reliable P(churn) scores for financial optimisation |
| **Engine** | Decision Optimisation | Expected Value Maximisation | Find Threshold\* that maximises net retention profit |

---

## Business Framing: The Economic Decision Theory Lens

### Why Standard Accuracy Metrics Fail in Banking

A naïve model optimising for **accuracy** on a 20% churn-rate dataset achieves 80% accuracy by predicting "no churn" for everyone — while destroying \$50M in recoverable lifetime value. The correct objective function is **Expected Profit**:

$$E[\text{Profit}(\tau)] = \sum_{i=1}^{N} \left[ \mathbf{1}(\hat{p}_i \geq \tau) \cdot \left( p_i \cdot \text{TP}_{\text{benefit}} + (1-p_i) \cdot \text{FP}_{\text{cost}} \right) + \mathbf{1}(\hat{p}_i < \tau) \cdot \left( p_i \cdot \text{FN}_{\text{cost}} + (1-p_i) \cdot \text{TN}_{\text{benefit}} \right) \right]$$

Where:
- $\tau$ = decision threshold
- $p_i$ = calibrated churn probability for customer $i$
- $\text{TP}_{\text{benefit}} = \text{CLV} - \text{RetentionCost}$ (saved customer net of intervention cost)
- $\text{FP}_{\text{cost}} = -\text{FalseAlarmCost}$ (loyalty offer wasted on a stayer)
- $\text{FN}_{\text{cost}} = -\text{CLV}$ (churner missed entirely)

**Threshold\*** is the $\tau$ that maximises this sum across the entire customer base.

---

## Architecture

```
customer-attrition-decision-engine/
│
├── data/
│   ├── raw/              ← Place Client.csv + Record.csv here
│   └── processed/        ← Auto-generated: merged_clean.parquet
│
├── notebooks/
│   └── OmnicampusAccountName.ipynb   ← Self-contained submission notebook
│
├── reports/
│   └── figures/          ← Auto-saved SHAP + Decision Engine plots (200 DPI)
│
├── src/
│   ├── config.py         ← Paths, constants, financial parameters (frozen dataclasses)
│   ├── data_loader.py    ← Schema validation, inner join, Parquet persistence
│   ├── features.py       ← 4 custom Scikit-learn Transformers + pipeline builder
│   ├── models.py         ← Stage-1 LightGBM + Stage-2 Stacking + Optuna search
│   ├── explainability.py ← SHAP Summary / Dependence / Waterfall plots
│   └── decision_engine.py← 2D Risk Matrix, Expected Profit Curve, Threshold*
│
├── requirements.txt
└── README.md
```

---

## The 2D Risk-Value Customer Matrix

After computing Threshold\*, each customer is placed in one of four strategic quadrants:

```
                    CLV
                High  │  Low
               ───────┼───────
    Churn   Hi │  Q2  │  Q3   │
    Risk       │ 🔴   │ 🟠   │
            ───┼──────┼───────┤
             Lo│  Q1  │  Q4   │
               │ 🟢   │ 🔵   │
               └──────┴───────┘
```

| Quadrant | Label | Strategy | Budget Priority |
|---|---|---|---|
| Q2 | **Priority Save** | High-touch retention, account manager escalation | ★★★★★ |
| Q3 | **Watch List** | Automated low-cost nudges, satisfaction survey | ★★★☆☆ |
| Q1 | **Champions** | Upsell, cross-sell, loyalty rewards | ★★★★☆ |
| Q4 | **Steady Base** | Maintain engagement, periodic check-in | ★★☆☆☆ |

---

## Feature Engineering

Four domain-driven features augment raw data:

| Feature | Formula | Intuition |
|---|---|---|
| `revenue_decay` | Balance / (Tenure + 1) | High balance + short tenure = acquisition risk |
| `usage_momentum` | NumProducts × IsActive | Product breadth with engagement decay |
| `engagement_score` | usage_momentum / (Tenure + 1) | Velocity of engagement decline |
| `friction_index` | Complain × (6 - Satisfaction) / (CreditScore/100) | Compound dissatisfaction signal |

OOF Target Encoding (K=5 folds) is applied to categorical features (`Geography`, `Gender`, `CardType`) to prevent **target leakage** — a critical requirement for financial model audits.

---

## Model Performance Targets

| Metric | Target |
|---|---|
| Stage-2 ROC-AUC | ≥ 0.88 |
| Brier Score | ≤ 0.10 |
| Expected Calibration Error | ≤ 0.03 |
| Lift @ Top 10% | ≥ 3.0× |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/<your-username>/customer-attrition-decision-engine.git
cd customer-attrition-decision-engine
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Place raw data
# → Copy Client.csv and Record.csv to data/raw/

# 3. Run the notebook
jupyter lab notebooks/OmnicampusAccountName.ipynb
```

---

## Data Sources

| Table | Rows | Key Columns | Join Key |
|---|---|---|---|
| `Client.csv` | ~100,000 | Age, Balance, Tenure, NumOfProducts, IsActiveMember, **Exited** | `Customer_ID` |
| `Record.csv` | ~100,000 | CreditScore, Complain, SatisfactionScore, CardType, PointEarned | `Customer_ID` |

> [!IMPORTANT]
> Raw data files are excluded from version control (`.gitignore`). Place `Client.csv` and `Record.csv` in `data/raw/` before running any code.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Wrangling | `pandas`, `numpy`, `scipy` |
| ML Framework | `scikit-learn`, `lightgbm`, `xgboost`, `catboost` |
| Hyperparameter Optimisation | `optuna` (TPE Sampler) |
| Probability Calibration | `sklearn.calibration.CalibratedClassifierCV` (Isotonic) |
| Explainability | `shap` (TreeExplainer + KernelExplainer fallback) |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Reproducibility | `watermark`, `RANDOM_STATE = 42` |

---

## Project Context

**Final Assignment** for [Course Name] — submitted as a fully reproducible Jupyter Notebook.

**Portfolio Purpose** — Demonstrates end-to-end Financial Data Science competency:
- Revenue-focused problem framing (beyond classification accuracy)
- Production-grade code organisation (`src/` package with type hints & docstrings)
- Leakage-free feature engineering (OOF Target Encoding)
- Calibrated probability outputs for financial decision making
- Expected Value Maximisation as a business objective function

---

## License

MIT License — see [LICENSE](LICENSE) for details.
