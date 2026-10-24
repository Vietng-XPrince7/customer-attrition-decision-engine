"""
explainability.py — SHAP-Based Explainable AI & Non-Linear Tipping Point Discovery
===================================================================================
Produces publication-quality (300 DPI) SHAP visualisations for the Dual-Stage
Attrition Engine using TreeExplainer on Gradient Boosted models.

Outputs:
- reports/figures/06_shap_summary_beeswarm.png
- reports/figures/07_shap_dependence_tipping_points.png

Usage:
    from src.explainability import ShapExplainer
    explainer = ShapExplainer(model=best_lgb, feature_names=feature_cols)
    explainer.compute_shap(X_test, sample_size=5000)
    explainer.plot_summary_beeswarm(save_path='reports/figures/06_shap_summary_beeswarm.png')
    explainer.plot_dependence_tipping_points(save_path='reports/figures/07_shap_dependence_tipping_points.png')
"""

import os, logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import PATHS, RANDOM_STATE

logger = logging.getLogger(__name__)


class ShapExplainer:
    """
    SHAP Interpretation Wrapper for Tree-Based Models.
    Optimized for large retail datasets with sub-sampling for fast, representative explanations.
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None
        self.X_sample = None

    def compute_shap(self, X: Union[pd.DataFrame, np.ndarray], sample_size: int = 5000) -> np.ndarray:
        """
        Compute SHAP values on a representative test sample.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            if len(X) > sample_size:
                self.X_sample = X.sample(n=sample_size, random_state=RANDOM_STATE)
            else:
                self.X_sample = X.copy()
            X_arr = self.X_sample.values
        else:
            if len(X) > sample_size:
                rng = np.random.RandomState(RANDOM_STATE)
                idx = rng.choice(len(X), size=sample_size, replace=False)
                X_arr = X[idx]
            else:
                X_arr = X
            self.X_sample = pd.DataFrame(X_arr, columns=self.feature_names)

        logger.info("Computing SHAP values for %d samples × %d features...", X_arr.shape[0], X_arr.shape[1])
        sv = self.explainer.shap_values(X_arr)
        if isinstance(sv, list) and len(sv) == 2:
            self.shap_values = sv[1]  # positive class (churn = 1)
        else:
            self.shap_values = sv

        return self.shap_values

    def plot_summary_beeswarm(
        self,
        max_display: int = 20,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Generate global feature importance bee-swarm plot.
        """
        if self.shap_values is None or self.X_sample is None:
            raise ValueError("Call compute_shap() before plotting.")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_size=(12, 8)
        )
        plt.title(
            f"SHAP Global Feature Importance (Bee-swarm) — Top {max_display} Drivers of Churn",
            fontsize=13, fontweight="bold", pad=15
        )
        plt.xlabel("SHAP Value (Impact on Log-Odds of Churn: >0 increases Churn risk)", fontsize=11)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("Saved SHAP summary beeswarm to %s", save_path)
        plt.close()

    def plot_dependence_tipping_points(
        self,
        primary_feature_1: str = "eqpdays",
        interaction_feature_1: str = "hnd_price",
        primary_feature_2: str = "rev_per_mou",
        interaction_feature_2: str = "totmrc_Mean",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Generate a 2-panel SHAP Dependence Plot revealing non-linear tipping points.
        """
        if self.shap_values is None or self.X_sample is None:
            raise ValueError("Call compute_shap() before plotting.")

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Panel 1: Device Obsolescence Tipping Point
        plt.sca(axes[0])
        shap.dependence_plot(
            primary_feature_1,
            self.shap_values,
            self.X_sample,
            interaction_index=interaction_feature_1 if interaction_feature_1 in self.X_sample.columns else None,
            ax=axes[0],
            show=False
        )
        axes[0].set_title(
            f"Physical Tipping Point: {primary_feature_1} vs SHAP Value\n(Color: {interaction_feature_1})",
            fontsize=12, fontweight="bold"
        )
        axes[0].set_ylabel("SHAP Value (Churn Impact)", fontsize=11)
        axes[0].axhline(0, color="black", linestyle="--", alpha=0.5, lw=1)
        axes[0].axvline(365, color="orange", linestyle=":", lw=1.5, label="1 Year Mark")
        axes[0].axvline(540, color="red", linestyle=":", lw=1.5, label="18-Month Danger Zone")
        axes[0].legend(loc="upper left")

        # Panel 2: Unit Cost Friction Tipping Point
        feat2 = primary_feature_2 if primary_feature_2 in self.X_sample.columns else "ovrrev_Mean"
        inter2 = interaction_feature_2 if interaction_feature_2 in self.X_sample.columns else "months"

        plt.sca(axes[1])
        shap.dependence_plot(
            feat2,
            self.shap_values,
            self.X_sample,
            interaction_index=inter2 if inter2 in self.X_sample.columns else None,
            ax=axes[1],
            show=False
        )
        axes[1].set_title(
            f"Financial Friction Point: {feat2} vs SHAP Value\n(Color: {inter2})",
            fontsize=12, fontweight="bold"
        )
        axes[1].set_ylabel("SHAP Value (Churn Impact)", fontsize=11)
        axes[1].axhline(0, color="black", linestyle="--", alpha=0.5, lw=1)

        fig.suptitle("Empirical Non-Linear Tipping Points Discovered by SHAP", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("Saved SHAP dependence plot to %s", save_path)
        plt.close()
