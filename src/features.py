"""
features.py — Leakage-Free Feature Engineering Pipeline (Phase 2)
==================================================================
Custom Scikit-learn Transformers for the Dual-Stage Attrition Engine.

Design Principles
-----------------
1. **Zero Leakage**: All statistics (medians, quantiles, target means) are
   computed exclusively on the training fold during ``fit()``; ``transform()``
   applies them to validation/test without re-computing.
2. **Robust to NaN**: Every transformer fills or flags missing values before
   producing engineered features.
3. **Winsorization-aware**: Extreme outliers are clipped at train-set quantiles,
   not global constants.
4. **Consistent API**: All transformers output a ``pd.DataFrame`` (not ndarray)
   preserving column names and index — compatible with ``ColumnTransformer``.

Transformers
------------
DomainFeatureGenerator      : 12 financial & behavioural engineered features
MissingnessAndOutlierHandler: Missing-value flags + imputation + Winsorization
OutOfFoldTargetEncoder      : K-Fold smoothed target encoding for high-cardinality cats

Pipeline Factory
----------------
build_leakage_free_pipeline : Assembles all three stages into a Pipeline

Usage
-----
    from src.features import build_leakage_free_pipeline

    pipe, oof_enc = build_leakage_free_pipeline()
    X_train_fe = pipe.fit_transform(X_train)
    X_train_fe = oof_enc.fit_transform(X_train_fe, y_train)
    X_test_fe  = pipe.transform(X_test)
    X_test_fe  = oof_enc.transform(X_test_fe)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.config import COLUMNS, RANDOM_STATE, CV_FOLDS

logger = logging.getLogger(__name__)

EPS: float = 1e-5   # avoid zero-division in ratio features


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Domain Feature Generator
# ═══════════════════════════════════════════════════════════════════════════════
class DomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate 12 financial & behavioural domain features.

    All arithmetic uses EPS guards to prevent division-by-zero.
    Missing values in source columns are filled with their training-set median
    (stored during ``fit``) before computing ratios.

    New Columns (12)
    ----------------
    Revenue & Usage Momentum
        rev_decay_3v6       avg3rev / (avg6rev + EPS)
        rev_decay_3vlife    avg3rev / (avgrev + EPS)
        mou_decay_3v6       avg3mou / (avg6mou + EPS)
        rev_per_mou         rev_Mean / (mou_Mean + 1.0)

    Bill Shock & Financial Friction
        overage_rev_ratio   ovrrev_Mean / (rev_Mean + EPS)
        fixed_charge_ratio  totmrc_Mean / (rev_Mean + EPS)
        active_sub_ratio    actvsubs / (uniqsubs + EPS)

    Network & Service Friction
        call_failure_rate   drop_blk_Mean / (attempt_Mean + 1.0)
        completion_rate     complete_Mean / (attempt_Mean + 1.0)
        custcare_intensity  custcare_Mean / (months + 1.0)

    Equipment Obsolescence
        eqp_burn_rate       eqpdays / (months * 30.0 + 1.0)
        handset_switch_freq phones / (months + 1.0)

    Parameters
    ----------
    eps : float
        Small constant added to denominators.
    clip_ratios : bool
        Clip all ratio outputs to [0, 10] to suppress extreme outliers.
    """

    def __init__(self, eps: float = EPS, clip_ratios: bool = True) -> None:
        self.eps         = eps
        self.clip_ratios = clip_ratios

    def fit(self, X: pd.DataFrame, y=None) -> "DomainFeatureGenerator":
        """Compute and store training-set medians for all source columns."""
        self._src_cols = [
            "avg3rev", "avg6rev", "avgrev", "avg3mou", "avg6mou",
            "rev_Mean", "mou_Mean", "ovrrev_Mean", "totmrc_Mean",
            "actvsubs", "uniqsubs", "drop_blk_Mean", "attempt_Mean",
            "complete_Mean", "custcare_Mean", "months",
            "eqpdays", "phones",
        ]
        self._medians: Dict[str, float] = {}
        for col in self._src_cols:
            if col in X.columns:
                self._medians[col] = float(X[col].median())
            else:
                self._medians[col] = 0.0
                logger.warning("DomainFeatureGenerator: '%s' not found; using 0.0", col)
        logger.info("DomainFeatureGenerator fitted on %d training rows.", len(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute all 12 domain features using training-set medians for NaN fill."""
        X = X.copy()

        def _get(col: str) -> pd.Series:
            """Return column filled with training-set median."""
            if col in X.columns:
                return X[col].fillna(self._medians.get(col, 0.0))
            return pd.Series(self._medians.get(col, 0.0), index=X.index)

        avg3rev   = _get("avg3rev");   avg6rev  = _get("avg6rev")
        avgrev    = _get("avgrev");    avg3mou  = _get("avg3mou")
        avg6mou   = _get("avg6mou");   rev_Mean = _get("rev_Mean")
        mou_Mean  = _get("mou_Mean");  ovrrev   = _get("ovrrev_Mean")
        totmrc    = _get("totmrc_Mean");actvsubs = _get("actvsubs")
        uniqsubs  = _get("uniqsubs");  drop_blk = _get("drop_blk_Mean")
        attempt   = _get("attempt_Mean"); complete = _get("complete_Mean")
        custcare  = _get("custcare_Mean"); months = _get("months")
        eqpdays   = _get("eqpdays");   phones   = _get("phones")

        e = self.eps

        # Revenue & Usage Momentum
        X["rev_decay_3v6"]    = avg3rev / (avg6rev + e)
        X["rev_decay_3vlife"] = avg3rev / (avgrev  + e)
        X["mou_decay_3v6"]    = avg3mou / (avg6mou + e)
        X["rev_per_mou"]      = rev_Mean / (mou_Mean + 1.0)

        # Bill Shock & Financial Friction
        X["overage_rev_ratio"]  = ovrrev  / (rev_Mean + e)
        X["fixed_charge_ratio"] = totmrc  / (rev_Mean + e)
        X["active_sub_ratio"]   = actvsubs / (uniqsubs + e)

        # Network & Service Friction
        X["call_failure_rate"]  = drop_blk / (attempt + 1.0)
        X["completion_rate"]    = complete  / (attempt + 1.0)
        X["custcare_intensity"] = custcare  / (months  + 1.0)

        # Equipment Obsolescence
        X["eqp_burn_rate"]       = eqpdays / (months * 30.0 + 1.0)
        X["handset_switch_freq"] = phones  / (months  + 1.0)

        if self.clip_ratios:
            for col in self.new_feature_names:
                cap = 50.0 if col == "rev_per_mou" else 10.0
                X[col] = X[col].clip(0, cap)

        logger.debug("DomainFeatureGenerator: added 12 features. Shape: %s", X.shape)
        return X

    @property
    def new_feature_names(self) -> List[str]:
        return [
            "rev_decay_3v6", "rev_decay_3vlife", "mou_decay_3v6", "rev_per_mou",
            "overage_rev_ratio", "fixed_charge_ratio", "active_sub_ratio",
            "call_failure_rate", "completion_rate", "custcare_intensity",
            "eqp_burn_rate", "handset_switch_freq",
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Missingness & Outlier Handler
# ═══════════════════════════════════════════════════════════════════════════════
class MissingnessAndOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Three-stage preprocessing:

    Stage A — Missing-value indicator flags
        Columns with > ``flag_threshold`` missing rate on training → binary
        ``{col}_is_missing`` column. Missingness in demographic/credit features
        carries independent churn-risk signal.

    Stage B — Imputation
        Numeric   : median  (training-set median)
        Categorical: constant label 'Missing'

    Stage C — Winsorization
        Clip numeric columns to [p01, p99] training-set quantiles.

    Parameters
    ----------
    flag_threshold : float
        Columns with > this fraction of NaNs get a missing indicator.
    winsor_low / winsor_high : float
        Quantiles for clipping.
    high_missing_cols : list, optional
        Columns that always receive a missing flag (known informative missingness).
    """

    KNOWN_INFORMATIVE_MISSING: List[str] = [
        "numbcars", "dwllsize", "income", "lor",
        "hnd_webcap", "HHstatin", "ownrent", "adults",
        "infobase", "dwlltype",
    ]

    def __init__(
        self,
        flag_threshold:     float = 0.05,
        winsor_low:         float = 0.01,
        winsor_high:        float = 0.99,
        high_missing_cols:  Optional[List[str]] = None,
    ) -> None:
        self.flag_threshold   = flag_threshold
        self.winsor_low       = winsor_low
        self.winsor_high      = winsor_high
        self.high_missing_cols = high_missing_cols

    def fit(self, X: pd.DataFrame, y=None) -> "MissingnessAndOutlierHandler":
        """Store flag columns, medians, and Winsorization bounds from training data."""
        # A: flag columns
        missing_rates = X.isnull().mean()
        auto_flag  = missing_rates[missing_rates > self.flag_threshold].index.tolist()
        force_flag = set(self.high_missing_cols or self.KNOWN_INFORMATIVE_MISSING)
        self.flag_cols_: List[str] = sorted(
            set(auto_flag) | (force_flag & set(X.columns))
        )
        logger.info("MissingnessHandler: %d flag columns.", len(self.flag_cols_))

        # B: medians
        self.num_cols_: List[str] = X.select_dtypes("number").columns.tolist()
        self.medians_: Dict[str, float] = {
            c: float(X[c].median()) for c in self.num_cols_
        }
        self.cat_cols_: List[str] = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # C: Winsorization quantiles
        self.lower_: Dict[str, float] = {}
        self.upper_: Dict[str, float] = {}
        for col in self.num_cols_:
            s = X[col].dropna()
            self.lower_[col] = float(s.quantile(self.winsor_low)) if len(s) else 0.0
            self.upper_[col] = float(s.quantile(self.winsor_high)) if len(s) else 0.0

        logger.info("MissingnessHandler fitted: %d num, %d cat cols.",
                    len(self.num_cols_), len(self.cat_cols_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply missing flags, imputation, and Winsorization."""
        X = X.copy()

        # A: Add missing-flag indicators
        for col in self.flag_cols_:
            if col in X.columns:
                X[f"{col}_is_missing"] = X[col].isnull().astype(np.int8)

        # B: Impute
        for col in self.num_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(self.medians_.get(col, 0.0))
        for col in self.cat_cols_:
            if col in X.columns:
                X[col] = X[col].fillna("Missing")

        # C: Winsorise
        for col in self.num_cols_:
            if col in X.columns:
                X[col] = X[col].clip(self.lower_.get(col), self.upper_.get(col))

        # Guard: kill any Inf values
        num_present = [c for c in self.num_cols_ if c in X.columns]
        X[num_present] = (X[num_present]
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(0.0))

        logger.debug("MissingnessHandler.transform: shape %s", X.shape)
        return X


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Out-of-Fold Target Encoder
# ═══════════════════════════════════════════════════════════════════════════════
class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leakage-free target encoding for high-cardinality categoricals.

    Training  (``fit_transform``): OOF K-Fold — validation rows encoded using
        stats from the other K-1 folds only → zero leakage.
    Inference (``transform``): uses full-train smoothed means from ``fit``.

    Smoothing: ``encoded = (n × col_mean + m × global_mean) / (n + m)``
    where m = ``smoothing`` (default 20 for telecom high-cardinality columns).

    Parameters
    ----------
    cols : list of str, optional
        Columns to encode. Defaults to DEFAULT_COLS.
    n_splits : int
    smoothing : float
    random_state : int
    drop_original : bool
        Drop the raw categorical column after adding the ``_te`` column.
    """

    DEFAULT_COLS: List[str] = [
        "crclscod", "area", "prizm_social_one", "ethnic",
        "dwllsize", "HHstatin", "ownrent", "dwlltype",
        "marital", "infobase",
    ]

    def __init__(
        self,
        cols:          Optional[List[str]] = None,
        n_splits:      int   = CV_FOLDS,
        smoothing:     float = 20.0,
        random_state:  int   = RANDOM_STATE,
        drop_original: bool  = False,
    ) -> None:
        self.cols          = cols
        self.n_splits      = n_splits
        self.smoothing     = smoothing
        self.random_state  = random_state
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OutOfFoldTargetEncoder":
        """Compute global means and per-category smoothed means on full training set."""
        self.cols_       = self.cols or [c for c in self.DEFAULT_COLS if c in X.columns]
        self.global_mean_ = float(y.mean())
        self.mapping_: Dict[str, Dict] = {}

        X = X.reset_index(drop=True)
        y = pd.Series(y.values, name="_target", index=X.index)

        for col in self.cols_:
            if col not in X.columns:
                logger.warning("OOFEncoder: '%s' not found; skipping.", col)
                continue
            stats = (
                pd.concat([X[col].astype(str), y], axis=1)
                .groupby(col, observed=True)
                .agg(count=(col, "count"), mean=("_target", "mean"))
            )
            m = self.smoothing
            self.mapping_[col] = (
                (stats["count"] * stats["mean"] + m * self.global_mean_)
                / (stats["count"] + m)
            ).to_dict()

        logger.info("OOFTargetEncoder fitted: %d columns.", len(self.mapping_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored training-set means (for val/test inference)."""
        X = X.copy()
        for col, mapping in self.mapping_.items():
            if col not in X.columns:
                continue
            X[f"{col}_te"] = (
                X[col].astype(str).map(mapping).fillna(self.global_mean_)
            )
        if self.drop_original:
            X = X.drop(columns=[c for c in self.mapping_ if c in X.columns])
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """OOF-aware fit_transform — zero leakage on training folds."""
        self.fit(X, y)   # stores global mapping for later transform() calls

        X = X.reset_index(drop=True).copy()
        y = pd.Series(y.values, name="_target", index=X.index)
        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)

        for col in self.cols_:
            if col not in X.columns:
                continue
            enc_col = f"{col}_te"
            X[enc_col] = self.global_mean_  # initialise with global mean

            for tr_idx, val_idx in kf.split(X):
                X_tr = X.iloc[tr_idx]; y_tr = y.iloc[tr_idx]
                stats = (
                    pd.concat([X_tr[col].astype(str), y_tr], axis=1)
                    .groupby(col, observed=True)
                    .agg(count=(col, "count"), mean=("_target", "mean"))
                )
                m = self.smoothing
                fold_map = (
                    (stats["count"] * stats["mean"] + m * self.global_mean_)
                    / (stats["count"] + m)
                ).to_dict()

                X.loc[val_idx, enc_col] = (
                    X.loc[val_idx, col].astype(str)
                    .map(fold_map).fillna(self.global_mean_).values
                )

        if self.drop_original:
            X = X.drop(columns=[c for c in self.cols_ if c in X.columns])

        logger.info("OOFTargetEncoder.fit_transform done: %s", self.cols_)
        return X


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Factory
# ═══════════════════════════════════════════════════════════════════════════════
def build_leakage_free_pipeline(
    target_encode_cols: Optional[List[str]] = None,
    smoothing:          float = 20.0,
    flag_threshold:     float = 0.05,
    drop_original_cats: bool  = False,
) -> Tuple["Pipeline", "OutOfFoldTargetEncoder"]:
    """
    Assemble the three-stage leakage-free feature engineering pipeline.

    Stage 1 : DomainFeatureGenerator       — 12 ratio features (no y needed)
    Stage 2 : MissingnessAndOutlierHandler  — flags + impute + winsorise
    Stage 3 : OutOfFoldTargetEncoder        — returned separately (needs y)

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Steps 1–2; call ``fit_transform(X)`` on train, ``transform(X)`` on test.
    oof_enc : OutOfFoldTargetEncoder
        Step 3; call ``fit_transform(X, y)`` on train, ``transform(X)`` on test.
    """
    pipeline = Pipeline([
        ("domain_features", DomainFeatureGenerator()),
        ("missingness",     MissingnessAndOutlierHandler(
                                flag_threshold=flag_threshold)),
    ])
    oof_enc = OutOfFoldTargetEncoder(
        cols=target_encode_cols,
        smoothing=smoothing,
        drop_original=drop_original_cats,
    )
    return pipeline, oof_enc


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Helper
# ═══════════════════════════════════════════════════════════════════════════════
def validate_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline test: fit on train, transform both splits.
    Raises AssertionError if NaN or Inf remain in numeric columns.

    Returns
    -------
    (X_train_fe, X_test_fe)
    """
    pipeline, oof_enc = build_leakage_free_pipeline()

    X_tr = pipeline.fit_transform(X_train)
    X_te = pipeline.transform(X_test)

    X_tr = oof_enc.fit_transform(X_tr, y_train)
    X_te = oof_enc.transform(X_te)

    for name, arr in [("Train", X_tr), ("Test", X_te)]:
        num   = arr.select_dtypes("number")
        n_nan = int(num.isnull().sum().sum())
        n_inf = int(np.isinf(num.values).sum())
        assert n_nan == 0, f"[{name}] NaN after pipeline: {n_nan} cells"
        assert n_inf == 0, f"[{name}] Inf after pipeline: {n_inf} cells"
        logger.info("[%s] Pipeline validation OK — shape %s", name, arr.shape)

    return X_tr, X_te
