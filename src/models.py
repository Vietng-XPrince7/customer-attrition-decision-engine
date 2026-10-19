"""
models.py — Dual-Stage Model Training Framework
=================================================
Implements the two-stage revenue defense architecture:

Stage 1 — Revenue Compression Detector
    Identifies customers experiencing revenue compression (downgrading,
    reducing balances) *before* full churn occurs.
    → LightGBM Regressor (predicted revenue loss) or Classifier (binary flag)

Stage 2 — Churn Probability Stacking Ensemble
    Combines LightGBM, XGBoost, and CatBoost base learners via a
    Logistic Regression meta-learner, with Optuna hyperparameter
    optimisation and Isotonic Regression probability calibration.

Usage
-----
    from src.models import Stage1RevenueModel, Stage2StackingEnsemble
    from src.models import run_optuna_study

    # Stage 2
    study  = run_optuna_study(X_train, y_train, n_trials=100)
    model  = Stage2StackingEnsemble(best_params=study.best_params)
    model.fit(X_train, y_train)
    proba  = model.predict_proba(X_test)[:, 1]
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from src.config import RANDOM_STATE, CV_FOLDS, OPTUNA_TRIALS, OPTUNA_TIMEOUT

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Stage 1 — Revenue Compression Model
# =============================================================================
class Stage1RevenueModel:
    """
    Stage-1 Revenue Compression Model.

    Detects customers experiencing revenue compression before they churn.
    Can operate in two modes:

    - ``mode='regressor'`` : Predicts continuous revenue loss amount (USD)
    - ``mode='classifier'``: Binary classification of revenue compression flag

    Parameters
    ----------
    mode : str
        Either ``'regressor'`` or ``'classifier'``.
    params : dict, optional
        LightGBM hyperparameters. Defaults to safe baseline values.
    random_state : int
    """

    def __init__(
        self,
        mode:         str = "classifier",
        params:       Optional[Dict[str, Any]] = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.mode         = mode
        self.random_state = random_state
        self.params       = params or self._default_params()
        self.model_       = self._build_model()

    def _default_params(self) -> Dict[str, Any]:
        base = dict(
            n_estimators    = 500,
            learning_rate   = 0.05,
            num_leaves      = 63,
            max_depth       = -1,
            min_child_samples = 20,
            subsample       = 0.8,
            colsample_bytree = 0.8,
            reg_alpha       = 0.1,
            reg_lambda      = 1.0,
            random_state    = self.random_state,
            n_jobs          = -1,
            verbose         = -1,
        )
        return base

    def _build_model(self):
        if self.mode == "regressor":
            return LGBMRegressor(**self.params)
        elif self.mode == "classifier":
            return LGBMClassifier(**self.params)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Use 'regressor' or 'classifier'.")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple] = None,
    ) -> "Stage1RevenueModel":
        """
        Fit the Stage-1 model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        eval_set : tuple (X_val, y_val), optional
            Validation set for early stopping.

        Returns
        -------
        self
        """
        fit_kwargs: Dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["callbacks"] = [
                optuna.integration.lightgbm.early_stopping(50, verbose=False)
                if hasattr(optuna, "integration") else None
            ]
            fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

        self.model_.fit(X, y, **fit_kwargs)
        logger.info("Stage1RevenueModel (%s) fitted.", self.mode)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "regressor":
            raise NotImplementedError("Use predict() for regressor mode.")
        return self.model_.predict_proba(X)

    def score_cv(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Return cross-validated ROC-AUC (classifier) or R² (regressor)."""
        metric = "roc_auc" if self.mode == "classifier" else "r2"
        scores = cross_val_score(
            self.model_, X, y,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=self.random_state)
            if self.mode == "classifier"
            else CV_FOLDS,
            scoring=metric,
            n_jobs=-1,
        )
        result = {"mean": float(scores.mean()), "std": float(scores.std()), "metric": metric}
        logger.info("Stage1 CV %s: %.4f ± %.4f", metric, result["mean"], result["std"])
        return result


# =============================================================================
# Stage 2 — Stacking Ensemble with Isotonic Calibration
# =============================================================================
class Stage2StackingEnsemble:
    """
    Stage-2 Churn Probability Stacking Ensemble.

    Base learners
    -------------
    - LightGBM Classifier
    - XGBoost Classifier
    - CatBoost Classifier

    Meta-learner
    ------------
    - Logistic Regression (regularised, fitted on OOF predictions)

    Post-processing
    ---------------
    - Isotonic Regression calibration via ``CalibratedClassifierCV``

    Parameters
    ----------
    lgb_params : dict, optional
        LightGBM hyperparameters.
    xgb_params : dict, optional
        XGBoost hyperparameters.
    cat_params : dict, optional
        CatBoost hyperparameters.
    calibrate : bool
        If True, wrap the stacking ensemble with Isotonic calibration.
    random_state : int
    """

    def __init__(
        self,
        lgb_params:   Optional[Dict[str, Any]] = None,
        xgb_params:   Optional[Dict[str, Any]] = None,
        cat_params:   Optional[Dict[str, Any]] = None,
        calibrate:    bool = True,
        random_state: int  = RANDOM_STATE,
    ) -> None:
        self.lgb_params   = lgb_params   or self._default_lgb()
        self.xgb_params   = xgb_params   or self._default_xgb()
        self.cat_params   = cat_params   or self._default_cat()
        self.calibrate    = calibrate
        self.random_state = random_state
        self.model_       = self._build()

    # ── Default hyperparameters ────────────────────────────────────────────────
    def _default_lgb(self) -> Dict[str, Any]:
        return dict(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=self.random_state, n_jobs=-1, verbose=-1,
        )

    def _default_xgb(self) -> Dict[str, Any]:
        return dict(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="logloss",
            random_state=self.random_state, n_jobs=-1, verbosity=0,
        )

    def _default_cat(self) -> Dict[str, Any]:
        return dict(
            iterations=500, learning_rate=0.05, depth=6,
            l2_leaf_reg=3, random_seed=self.random_state,
            verbose=False, allow_writing_files=False,
        )

    def _build(self):
        estimators = [
            ("lgb", LGBMClassifier(**self.lgb_params)),
            ("xgb", XGBClassifier(**self.xgb_params)),
            ("cat", CatBoostClassifier(**self.cat_params)),
        ]
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=self.random_state),
            passthrough=False,
            n_jobs=-1,
        )
        if self.calibrate:
            return CalibratedClassifierCV(
                estimator=stack,
                method="isotonic",
                cv=CV_FOLDS,
            )
        return stack

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Stage2StackingEnsemble":
        """
        Fit the stacking ensemble (and calibration wrapper if enabled).

        Parameters
        ----------
        X : array-like
        y : array-like (binary 0/1)

        Returns
        -------
        self
        """
        logger.info("Fitting Stage2StackingEnsemble (calibrate=%s)...", self.calibrate)
        self.model_.fit(X, y)
        logger.info("Stage2StackingEnsemble fitted.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics on a held-out set.

        Returns
        -------
        dict with keys: roc_auc, average_precision, log_loss
        """
        proba = self.predict_proba(X)[:, 1]
        metrics = {
            "roc_auc":           roc_auc_score(y, proba),
            "average_precision": average_precision_score(y, proba),
            "log_loss":          log_loss(y, proba),
        }
        for k, v in metrics.items():
            logger.info("Stage2 %s: %.4f", k, v)
        return metrics


# =============================================================================
# Optuna Hyperparameter Optimisation
# =============================================================================
def _lgb_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> float:
    """Optuna objective for LightGBM — maximises CV ROC-AUC."""
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 20, 300),
        "max_depth":          trial.suggest_int("max_depth", 3, 12),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 100),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "class_weight":       "balanced",
        "random_state":       random_state,
        "n_jobs":             -1,
        "verbose":            -1,
    }
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def run_optuna_study(
    X: np.ndarray,
    y: np.ndarray,
    n_trials:     int = OPTUNA_TRIALS,
    timeout:      int = OPTUNA_TIMEOUT,
    study_name:   str = "lgb_churn_optimisation",
    direction:    str = "maximize",
) -> optuna.Study:
    """
    Launch an Optuna hyperparameter search for the LightGBM base learner.

    Parameters
    ----------
    X : array-like
    y : array-like (binary)
    n_trials : int
        Maximum number of Optuna trials.
    timeout : int
        Maximum search duration in seconds.
    study_name : str
    direction : str
        ``'maximize'`` for ROC-AUC.

    Returns
    -------
    optuna.Study
        Completed study; access best params via ``study.best_params``.
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
    )
    study.optimize(
        lambda trial: _lgb_objective(trial, X, y),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )
    logger.info(
        "Optuna complete. Best ROC-AUC: %.4f | Best params: %s",
        study.best_value, study.best_params,
    )
    return study

