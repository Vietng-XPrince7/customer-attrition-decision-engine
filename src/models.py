"""
models.py — Dual-Stage Model Training, Optuna Tuning & Probability Calibration
================================================================================
Implements:
1. **Stage 1 — Early Warning Model (Revenue Compression / Silent Attrition)**:
   - Target: `severe_compression` (change_rev < -10.0 or rev_decay_3v6 < 0.85)
   - LEAKAGE PREVENTION: Automatically drops direct target leakage columns:
     ['change_rev', 'avg3rev', 'rev_decay_3v6', 'rev_decay_3vlife', 'change_mou', 'mou_decay_3v6']
   - LightGBM Classifier with class-balanced weighting.

2. **Stage 2 — Churn Prediction Benchmark & Soft Voting / Stacking Ensemble**:
   - Compares: LogisticRegression (Baseline), RandomForest, XGBoost, CatBoost, Tuned LightGBM.
   - Optuna Bayesian Search (25-30 trials) optimizing ROC-AUC for LightGBM.
   - Ensemble combining Top 3: Tuned LightGBM + CatBoost + XGBoost.
   - Isotonic Probability Calibration via CalibratedClassifierCV.
   - Publication benchmark table & 4-panel diagnostic plot:
     reports/figures/05_model_benchmark_and_calibration.png (300 DPI)
"""

import os, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import catboost as cb
from catboost import CatBoostClassifier
import optuna

from src.config import RANDOM_STATE, CV_FOLDS

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Stage 1 — Early Warning Model (Revenue Compression)
# ═══════════════════════════════════════════════════════════════════════════════
class Stage1EarlyWarningModel:
    """
    Predicts early customer revenue compression without target leakage.
    """

    LEAKY_COLUMNS: List[str] = [
        "change_rev", "avg3rev", "rev_decay_3v6", "rev_decay_3vlife",
        "change_mou", "mou_decay_3v6", "avg3qty", "avg3mou",
    ]

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state
        self.model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        self.feature_names_: List[str] = []

    def _sanitize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in self.LEAKY_COLUMNS if c in X.columns]
        if drop_cols:
            logger.info("Stage 1: Dropped %d leakage columns: %s", len(drop_cols), drop_cols)
        return X.drop(columns=drop_cols)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "Stage1EarlyWarningModel":
        X_clean = self._sanitize_features(X_train)
        self.feature_names_ = X_clean.columns.tolist()
        logger.info("Fitting Stage 1 Early Warning Model on %d clean features...", len(self.feature_names_))
        self.model.fit(X_clean, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = self._sanitize_features(X)
        return self.model.predict_proba(X_clean)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        proba = self.predict_proba(X_test)
        pred = (proba >= 0.50).astype(int)
        metrics = {
            "ROC-AUC": float(roc_auc_score(y_test, proba)),
            "PR-AUC": float(average_precision_score(y_test, proba)),
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, pred, zero_division=0)),
            "Brier": float(brier_score_loss(y_test, proba)),
        }
        logger.info("Stage 1 Early Warning Metrics: %s", metrics)
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Stage 2 — Optuna Hyperparameter Optimization for LightGBM
# ═══════════════════════════════════════════════════════════════════════════════
def tune_lgbm_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 25,
    timeout: int = 120,
    random_state: int = RANDOM_STATE
) -> Tuple[Dict[str, Any], float]:
    """
    Tune LightGBM hyperparameters on training set using a dedicated stratified validation split.
    Fast, reliable, and immune to Windows multiprocessing deadlocks.
    """
    from sklearn.model_selection import train_test_split
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=random_state
    )

    logger.info("Starting Optuna tuning for LightGBM (%d trials on %d sub-train / %d val)...",
                n_trials, len(X_tr_sub), len(X_val_sub))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 70),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_tr_sub, y_tr_sub)
        proba = model.predict_proba(X_val_sub)[:, 1]
        return float(roc_auc_score(y_val_sub, proba))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info("Optuna best trial: Val ROC-AUC = %.4f | Params: %s", study.best_value, study.best_params)
    return study.best_params, study.best_value


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Model Benchmark Suite & Calibration
# ═══════════════════════════════════════════════════════════════════════════════
class ModelBenchmarkSuite:
    """
    Trains and benchmarks 5 models + 1 Ensemble + 1 Calibrated Ensemble.
    """

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.calibrated_ensemble = None

    def fit_and_benchmark(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        lgb_best_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Train all models, calibrate the ensemble, and produce the benchmark report.
        """
        logger.info("Initializing models for benchmark...")

        # 1. Baseline Logistic Regression (with Scaling)
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=self.random_state))
        ])

        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=10,
            n_jobs=-1, random_state=self.random_state
        )

        # 3. XGBoost
        xgb_model = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=self.random_state
        )

        # 4. CatBoost
        cat_model = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            verbose=False, random_seed=self.random_state
        )

        # 5. LightGBM (Tuned or Default)
        if lgb_best_params:
            lgb_params = {**lgb_best_params, "random_state": self.random_state, "n_jobs": -1, "verbose": -1}
        else:
            lgb_params = {
                "n_estimators": 250, "learning_rate": 0.05, "num_leaves": 45,
                "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
                "random_state": self.random_state, "n_jobs": -1, "verbose": -1
            }
        lgb_model = LGBMClassifier(**lgb_params)

        # Fit individual models
        self.models = {
            "Logistic Regression": lr_pipe,
            "Random Forest": rf,
            "XGBoost": xgb_model,
            "CatBoost": cat_model,
            "LightGBM (Tuned)": lgb_model,
        }

        for name, m in self.models.items():
            logger.info("Fitting %s...", name)
            m.fit(X_train, y_train)

        # 6. Soft Voting Ensemble (Top 3: LightGBM + CatBoost + XGBoost)
        logger.info("Fitting Soft Voting Ensemble (LGB + CatBoost + XGB)...")
        ensemble = VotingClassifier(
            estimators=[
                ("lgb", lgb_model),
                ("cat", cat_model),
                ("xgb", xgb_model),
            ],
            voting="soft",
            n_jobs=1
        )
        # Note: Base models are already fitted! We set fitted estimators directly or fit quickly
        ensemble.fit(X_train, y_train)
        self.models["Ensemble (LGB+Cat+XGB)"] = ensemble

        # 7. Probability Calibration (Isotonic via prefit calibration split)
        logger.info("Applying Isotonic Probability Calibration to Ensemble...")
        from sklearn.model_selection import train_test_split
        _, X_cal, _, y_cal = train_test_split(
            X_train, y_train, test_size=0.20, stratify=y_train, random_state=self.random_state
        )
        try:
            from sklearn.frozen import FrozenEstimator
            calibrated_ens = CalibratedClassifierCV(
                estimator=FrozenEstimator(ensemble),
                method="isotonic"
            )
        except (ImportError, Exception):
            calibrated_ens = CalibratedClassifierCV(
                estimator=ensemble,
                method="isotonic",
                cv=2
            )
        calibrated_ens.fit(X_cal, y_cal)
        self.calibrated_ensemble = calibrated_ens
        self.models["Ensemble (Calibrated)"] = calibrated_ens

        # Compute benchmark metrics
        records = []
        for name, model in self.models.items():
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.50).astype(int)

            rec = {
                "Model": name,
                "ROC-AUC": roc_auc_score(y_test, proba),
                "PR-AUC": average_precision_score(y_test, proba),
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1-Score": f1_score(y_test, pred, zero_division=0),
                "Brier Score": brier_score_loss(y_test, proba),
            }
            records.append(rec)

        benchmark_df = pd.DataFrame(records).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
        return benchmark_df

    def plot_benchmark_and_calibration(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Generate 4-panel diagnostic plot: ROC curves, PR curves, Reliability diagram, Probability distribution.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Color mapping
        colors = {
            "Logistic Regression": "#78909C",
            "Random Forest": "#AB47BC",
            "XGBoost": "#FB8C00",
            "CatBoost": "#00ACC1",
            "LightGBM (Tuned)": "#1E88E5",
            "Ensemble (LGB+Cat+XGB)": "#E53935",
            "Ensemble (Calibrated)": "#2E7D32",
        }

        # 1. ROC Curves
        ax_roc = axes[0, 0]
        for name, m in self.models.items():
            proba = m.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            lw = 2.5 if "Ensemble" in name else 1.5
            ls = "--" if name == "Ensemble (Calibrated)" else "-"
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=colors.get(name, "black"), lw=lw, linestyle=ls)

        ax_roc.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.6)
        ax_roc.set_title("Receiver Operating Characteristic (ROC) Benchmark", fontsize=12, fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate", fontsize=11)
        ax_roc.legend(loc="lower right", fontsize=8.5)
        ax_roc.grid(True, alpha=0.3, linestyle="--")

        # 2. Precision-Recall Curves
        ax_pr = axes[0, 1]
        for name, m in self.models.items():
            proba = m.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, proba)
            ap = average_precision_score(y_test, proba)
            lw = 2.5 if "Ensemble" in name else 1.5
            ls = "--" if name == "Ensemble (Calibrated)" else "-"
            ax_pr.plot(rec, prec, label=f"{name} (PR-AUC={ap:.4f})", color=colors.get(name, "black"), lw=lw, linestyle=ls)

        ax_pr.set_title("Precision-Recall Benchmark Curve", fontsize=12, fontweight="bold")
        ax_pr.set_xlabel("Recall", fontsize=11)
        ax_pr.set_ylabel("Precision", fontsize=11)
        ax_pr.legend(loc="lower left", fontsize=8.5)
        ax_pr.grid(True, alpha=0.3, linestyle="--")

        # 3. Calibration Curve / Reliability Diagram
        ax_cal = axes[1, 0]
        ens_raw = self.models["Ensemble (LGB+Cat+XGB)"]
        ens_cal = self.calibrated_ensemble

        proba_raw = ens_raw.predict_proba(X_test)[:, 1]
        proba_cal = ens_cal.predict_proba(X_test)[:, 1]

        prob_true_raw, prob_pred_raw = calibration_curve(y_test, proba_raw, n_bins=10, strategy="uniform")
        prob_true_cal, prob_pred_cal = calibration_curve(y_test, proba_cal, n_bins=10, strategy="uniform")

        brier_raw = brier_score_loss(y_test, proba_raw)
        brier_cal = brier_score_loss(y_test, proba_cal)

        ax_cal.plot(prob_pred_raw, prob_true_raw, "s-", color="#E53935", lw=2, label=f"Uncalibrated Ensemble (Brier={brier_raw:.4f})")
        ax_cal.plot(prob_pred_cal, prob_true_cal, "o-", color="#2E7D32", lw=2.5, label=f"Isotonic Calibrated (Brier={brier_cal:.4f})")
        ax_cal.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Reliability (Identity)")

        ax_cal.set_title("Reliability Diagram: Pre vs. Post Probability Calibration", fontsize=12, fontweight="bold")
        ax_cal.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax_cal.set_ylabel("Empirical Fraction of Positives", fontsize=11)
        ax_cal.legend(loc="upper left", fontsize=9)
        ax_cal.grid(True, alpha=0.3, linestyle="--")

        # 4. Calibrated Probability Distribution
        ax_dist = axes[1, 1]
        ax_dist.hist(proba_cal[y_test == 0], bins=35, alpha=0.55, color="#1565C0", label="Stayers (y=0)", density=True)
        ax_dist.hist(proba_cal[y_test == 1], bins=35, alpha=0.55, color="#E53935", label="Churners (y=1)", density=True)
        ax_dist.axvline(0.50, color="black", linestyle="--", lw=1.5, label="Default Cutoff (p=0.5)")

        ax_dist.set_title("Calibrated Churn Probability Density by Actual Class", fontsize=12, fontweight="bold")
        ax_dist.set_xlabel("Calibrated P(Churn)", fontsize=11)
        ax_dist.set_ylabel("Probability Density", fontsize=11)
        ax_dist.legend(loc="upper center", fontsize=9)
        ax_dist.grid(True, alpha=0.3, linestyle="--")

        fig.suptitle("Stage 2 — Model Benchmark Suite & Isotonic Calibration Analysis", fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("Saved benchmark and calibration plot to %s", save_path)
        plt.close()
