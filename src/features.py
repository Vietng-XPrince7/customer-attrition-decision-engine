"""
features.py — Domain-Driven Feature Engineering Pipeline
=========================================================
Custom Scikit-learn compatible transformers implementing four core
financial feature concepts:

1. **RevenuDecayTransformer**   — Captures revenue-at-risk via balance × tenure decay
2. **UsageMomentumTransformer** — Measures product engagement velocity
3. **FrictionIndexTransformer** — Composite dissatisfaction signal
4. **OutOfFoldTargetEncoder**   — Leakage-free target encoding using K-Fold OOF

All transformers follow the ``BaseEstimator`` + ``TransformerMixin`` contract
and are compatible with ``sklearn.pipeline.Pipeline``.

Usage
-----
    from src.features import build_feature_pipeline
    pipeline = build_feature_pipeline()
    X_transformed = pipeline.fit_transform(X_train, y_train)
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.config import COLUMNS, RANDOM_STATE, CV_FOLDS

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Revenue Decay Transformer
# =============================================================================
class RevenuDecayTransformer(BaseEstimator, TransformerMixin):
    """
    Compute a Revenue Decay feature capturing how much wealth is tied up
    in the bank relative to how long the customer has been present.

    Feature formula
    ---------------
    ``revenue_decay = Balance / (Tenure + 1)``

    A customer with a high balance but very short tenure signals either
    recent large deposit (acquisition risk) or a churner consolidating funds.
    """

    def __init__(
        self,
        balance_col: str = "Balance",
        tenure_col:  str = "Tenure",
    ) -> None:
        self.balance_col = balance_col
        self.tenure_col  = tenure_col

    def fit(self, X: pd.DataFrame, y=None) -> "RevenuDecayTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``balance_col`` and ``tenure_col``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with an additional ``revenue_decay`` column.
        """
        X = X.copy()
        X["revenue_decay"] = X[self.balance_col] / (X[self.tenure_col] + 1)
        logger.debug("RevenuDecayTransformer: added 'revenue_decay' column.")
        return X


# =============================================================================
# 2. Usage Momentum Transformer
# =============================================================================
class UsageMomentumTransformer(BaseEstimator, TransformerMixin):
    """
    Capture engagement velocity as the interaction between product breadth
    and active membership status.

    Feature formula
    ---------------
    ``usage_momentum = NumOfProducts × IsActiveMember``
    ``engagement_score = (NumOfProducts × IsActiveMember) / (Tenure + 1)``

    A customer with many products but now inactive is at high churn risk.
    """

    def __init__(
        self,
        products_col:   str = "NumOfProducts",
        active_col:     str = "IsActiveMember",
        tenure_col:     str = "Tenure",
    ) -> None:
        self.products_col = products_col
        self.active_col   = active_col
        self.tenure_col   = tenure_col

    def fit(self, X: pd.DataFrame, y=None) -> "UsageMomentumTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``usage_momentum`` and ``engagement_score`` columns.
        """
        X = X.copy()
        X["usage_momentum"]   = X[self.products_col] * X[self.active_col]
        X["engagement_score"] = X["usage_momentum"] / (X[self.tenure_col] + 1)
        logger.debug("UsageMomentumTransformer: added engagement features.")
        return X


# =============================================================================
# 3. Friction Index Transformer
# =============================================================================
class FrictionIndexTransformer(BaseEstimator, TransformerMixin):
    """
    Construct a Composite Friction Index — a single dissatisfaction signal
    combining complaint history, satisfaction score, and credit health.

    Feature formula
    ---------------
    ``friction_index = Complain × (6 - SatisfactionScore) / (CreditScore / 100)``

    Higher friction index → higher churn propensity driven by bad experience
    compounded by financial stress.
    """

    def __init__(
        self,
        complain_col:     str = "Complain",
        satisfaction_col: str = "SatisfactionScore",
        credit_col:       str = "CreditScore",
    ) -> None:
        self.complain_col     = complain_col
        self.satisfaction_col = satisfaction_col
        self.credit_col       = credit_col

    def fit(self, X: pd.DataFrame, y=None) -> "FrictionIndexTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``friction_index`` column added.
        """
        X = X.copy()
        numerator   = X[self.complain_col] * (6 - X[self.satisfaction_col])
        denominator = X[self.credit_col] / 100 + 1e-6   # avoid division by zero
        X["friction_index"] = numerator / denominator
        logger.debug("FrictionIndexTransformer: added 'friction_index' column.")
        return X


# =============================================================================
# 4. Out-of-Fold Target Encoder
# =============================================================================
class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leakage-free target encoding using Out-of-Fold (OOF) K-Fold averages.

    For training data: each fold uses the mean target computed on all *other*
    folds, preventing target leakage.
    For test/validation data: uses the global mean computed on the full
    training set (stored in ``fit()``).

    Parameters
    ----------
    cols : list of str
        Categorical columns to encode.
    n_splits : int
        Number of K-Fold splits.
    smoothing : float
        Smoothing factor (additive, Bayesian-style).
        ``encoded = (count × col_mean + smoothing × global_mean) / (count + smoothing)``
    random_state : int
    """

    def __init__(
        self,
        cols:         List[str],
        n_splits:     int   = CV_FOLDS,
        smoothing:    float = 10.0,
        random_state: int   = RANDOM_STATE,
    ) -> None:
        self.cols         = cols
        self.n_splits     = n_splits
        self.smoothing    = smoothing
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OutOfFoldTargetEncoder":
        """
        Compute and store per-category target means on the full training set.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series
            Binary or continuous target vector.

        Returns
        -------
        self
        """
        self.global_mean_: float = float(y.mean())
        self.mapping_: dict[str, dict] = {}

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True) if hasattr(y, "reset_index") else pd.Series(y)

        for col in self.cols:
            stats = (
                pd.concat([X[col], y], axis=1)
                .groupby(col)
                .agg(count=(col, "count"), mean=(y.name or 0, "mean"))
            )
            smooth = (
                (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
                / (stats["count"] + self.smoothing)
            )
            self.mapping_[col] = smooth.to_dict()

        logger.info("OutOfFoldTargetEncoder fitted on cols: %s", self.cols)
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply smoothed target encoding.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series, optional
            If provided (training phase), OOF encoding is applied.
            If None (inference), stored training means are used.

        Returns
        -------
        pd.DataFrame
        """
        X = X.copy().reset_index(drop=True)
        for col in self.cols:
            enc_col = f"{col}_te"
            X[enc_col] = (
                X[col]
                .map(self.mapping_[col])
                .fillna(self.global_mean_)
            )
        logger.debug("OutOfFoldTargetEncoder: encoded columns %s", self.cols)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """OOF-aware fit_transform: applies fold-level encoding to avoid leakage."""
        self.fit(X, y)
        X = X.copy().reset_index(drop=True)
        y = y.reset_index(drop=True) if hasattr(y, "reset_index") else pd.Series(y)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        oof_encoded = X.copy()

        for col in self.cols:
            enc_col = f"{col}_te"
            oof_encoded[enc_col] = self.global_mean_   # default

            for train_idx, val_idx in kf.split(X):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                stats = (
                    pd.concat([X_tr[col], y_tr], axis=1)
                    .groupby(col)
                    .agg(count=(col, "count"), mean=(y.name or 0, "mean"))
                )
                smooth = (
                    (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
                    / (stats["count"] + self.smoothing)
                )
                oof_encoded.loc[val_idx, enc_col] = (
                    X.loc[val_idx, col]
                    .map(smooth.to_dict())
                    .fillna(self.global_mean_)
                    .values
                )

        return oof_encoded


# =============================================================================
# Pipeline Builder
# =============================================================================
def build_feature_pipeline(
    numeric_cols:      Optional[List[str]] = None,
    categorical_cols:  Optional[List[str]] = None,
) -> Pipeline:
    """
    Assemble the full Scikit-learn feature engineering pipeline.

    Steps
    -----
    1. RevenuDecayTransformer
    2. UsageMomentumTransformer
    3. FrictionIndexTransformer
    4. ColumnTransformer:
       - Numeric → Impute (median) → StandardScaler
       - Categorical → Impute (most_frequent) → OrdinalEncoder

    Parameters
    ----------
    numeric_cols : list of str, optional
        Numeric feature names. Defaults to config-derived list.
    categorical_cols : list of str, optional
        Categorical feature names. Defaults to ``COLUMNS.categorical``.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if numeric_cols is None:
        numeric_cols = [
            "Age", "Tenure", "EstimatedSalary", "Balance",
            "NumOfProducts", "CreditScore", "SatisfactionScore",
            "PointEarned", "revenue_decay", "usage_momentum",
            "engagement_score", "friction_index",
        ]
    if categorical_cols is None:
        categorical_cols = COLUMNS.categorical

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipeline,     numeric_cols),
            ("cat",  categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("revenue_decay",     RevenuDecayTransformer()),
        ("usage_momentum",    UsageMomentumTransformer()),
        ("friction_index",    FrictionIndexTransformer()),
        ("col_transformer",   col_transformer),
    ])

    logger.info("Feature pipeline assembled with %d numeric and %d categorical features.",
                len(numeric_cols), len(categorical_cols))
    return pipeline

