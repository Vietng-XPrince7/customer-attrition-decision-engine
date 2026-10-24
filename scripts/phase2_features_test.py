"""
Phase 2 — Feature Engineering Pipeline Validation & Driver Analysis
====================================================================
Tests src.features pipeline end-to-end:
1. Leakage-free fit on train, transform on test.
2. NaN & Inf integrity checks.
3. Shape inspection.
4. Top 15 Spearman & Mutual Information rankings for churn & change_rev.
"""

import sys, warnings, os
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from src.config import RANDOM_STATE, COLUMNS
from src.features import build_leakage_free_pipeline, validate_pipeline

print("=" * 70)
print("  PHASE 2 — FEATURE ENGINEERING PIPELINE VALIDATION")
print("=" * 70)

df = pd.read_parquet(ROOT / "data/processed/merged_clean.parquet")
print(f"[DATA] Loaded {len(df):,} rows × {df.shape[1]} cols")

target = df["churn"].astype(int)
features = df.drop(columns=["Customer_ID", "churn"], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.20, stratify=target, random_state=RANDOM_STATE
)
print(f"[SPLIT] Train: {X_train.shape} | Test: {X_test.shape}")

# Run validation pipeline
print("\n[PIPELINE] Running build_leakage_free_pipeline()...")
pipeline, oof_enc = build_leakage_free_pipeline(flag_threshold=0.10)

# Step 1 & 2: Domain Features + Missingness/Outlier Handler
X_tr_p1 = pipeline.fit_transform(X_train)
X_te_p1 = pipeline.transform(X_test)
print(f"[STAGE 1+2] After Domain Features + Imputation + Winsorization:")
print(f"  Train: {X_tr_p1.shape} | Test: {X_te_p1.shape}")

# Step 3: OOF Target Encoding
X_tr_fe = oof_enc.fit_transform(X_tr_p1, y_train)
X_te_fe = oof_enc.transform(X_te_p1)
print(f"[STAGE 3] After Out-Of-Fold Target Encoding:")
print(f"  Train: {X_tr_fe.shape} | Test: {X_te_fe.shape}")

# Integrity checks
num_tr = X_tr_fe.select_dtypes("number")
num_te = X_te_fe.select_dtypes("number")
nan_tr = num_tr.isnull().sum().sum()
nan_te = num_te.isnull().sum().sum()
inf_tr = np.isinf(num_tr.values).sum()
inf_te = np.isinf(num_te.values).sum()

print("\n[INTEGRITY CHECKS]")
print(f"  Train NaN count: {nan_tr} | Inf count: {inf_tr}")
print(f"  Test  NaN count: {nan_te} | Inf count: {inf_te}")
assert nan_tr == 0 and nan_te == 0, "Error: NaN detected in numeric columns!"
assert inf_tr == 0 and inf_te == 0, "Error: Inf detected in numeric columns!"
print("  >>> INTEGRITY VALIDATION PASSED: 100% clean of NaN and Inf! <<<")

# Check new domain features
new_domain_cols = [
    "rev_decay_3v6", "rev_decay_3vlife", "mou_decay_3v6", "rev_per_mou",
    "overage_rev_ratio", "fixed_charge_ratio", "active_sub_ratio",
    "call_failure_rate", "completion_rate", "custcare_intensity",
    "eqp_burn_rate", "handset_switch_freq",
]
print("\n[DOMAIN FEATURES VERIFICATION]")
for c in new_domain_cols:
    s = X_tr_fe[c]
    print(f"  {c:<22}: min={s.min():.4f}, median={s.median():.4f}, max={s.max():.4f}")

# Check missingness indicators
missing_flags = [c for c in X_tr_fe.columns if c.endswith("_is_missing")]
print(f"\n[MISSINGNESS FLAGS GENERATED ({len(missing_flags)})]")
print("  Flags:", missing_flags)

# Target Encoded columns
te_cols = [c for c in X_tr_fe.columns if c.endswith("_te")]
print(f"\n[TARGET ENCODED COLUMNS ({len(te_cols)})]")
print("  TE cols:", te_cols)

# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION & IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  TOP 15 CORRELATIONS WITH CHURN")
print("─" * 70)

spear_churn = {}
for col in num_tr.columns:
    try:
        r, p = stats.spearmanr(num_tr[col], y_train)
        if not np.isnan(r):
            spear_churn[col] = r
    except Exception:
        pass

s_churn = pd.Series(spear_churn)
s_churn_abs = s_churn.abs().sort_values(ascending=False)

for i, (col, r_abs) in enumerate(s_churn_abs.head(15).items(), 1):
    r_val = s_churn[col]
    dir_str = "(+ Churn risk)" if r_val > 0 else "(- Protective)"
    print(f"  {i:2d}. {col:<28} |r|={r_abs:.4f}  (r={r_val:+.4f}) {dir_str}")

print("\n" + "─" * 70)
print("  TOP 15 CORRELATIONS WITH CHANGE_REV (REVENUE COMPRESSION)")
print("─" * 70)

if "change_rev" in num_tr.columns:
    target_rev = num_tr["change_rev"]
    spear_rev = {}
    for col in num_tr.columns:
        if col == "change_rev":
            continue
        try:
            r, p = stats.spearmanr(num_tr[col], target_rev)
            if not np.isnan(r):
                spear_rev[col] = r
        except Exception:
            pass

    s_rev = pd.Series(spear_rev)
    s_rev_abs = s_rev.abs().sort_values(ascending=False)
    for i, (col, r_abs) in enumerate(s_rev_abs.head(15).items(), 1):
        r_val = s_rev[col]
        print(f"  {i:2d}. {col:<28} |r|={r_abs:.4f}  (r={r_val:+.4f})")

print("\n" + "─" * 70)
print("  MUTUAL INFORMATION WITH CHURN (Sample of 20,000)")
print("─" * 70)

sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(num_tr), size=20000, replace=False)
X_sub = num_tr.iloc[sample_idx]
y_sub = y_train.iloc[sample_idx]

mi_scores = mutual_info_classif(X_sub, y_sub, random_state=RANDOM_STATE, n_neighbors=5)
mi_series = pd.Series(mi_scores, index=num_tr.columns).sort_values(ascending=False)

for i, (col, mi) in enumerate(mi_series.head(15).items(), 1):
    print(f"  {i:2d}. {col:<28} MI={mi:.4f}")

print("\n" + "=" * 70)
print("  FEATURE ENGINEERING VALIDATION COMPLETE")
print("=" * 70)
