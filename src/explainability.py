"""
explainability.py — SHAP-Based Model Interpretation
=====================================================
Generates publication-quality SHAP visualisations for the Dual-Stage
attrition models, supporting:

- Summary Plot  (global feature importance bee-swarm)
- Dependence Plot (feature interaction analysis)
- Waterfall Plot  (local, per-customer explanation)
- Force Plot      (HTML-embeddable individual explanation)

All figures are saved to ``PATHS.figures`` in high-resolution PNG.

Usage
-----
    from src.explainability import ShapExplainer

    explainer = ShapExplainer(model=stage2_model, X_background=X_train_sample)
    explainer.summary_plot(X_test)
    explainer.dependence_plot(X_test, feature="friction_index")
    explainer.waterfall_plot(X_test, row_idx=0)
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import PATHS, SHAP_MAX_DISPLAY, SHAP_SAMPLE_SIZE, RANDOM_STATE

logger = logging.getLogger(__name__)


# =============================================================================
# SHAP Explainer Wrapper
# =============================================================================
class ShapExplainer:
    """
    Unified SHAP explainer wrapper for tree-based and calibrated ensemble models.

    Automatically selects the appropriate SHAP explainer type:
    - ``shap.TreeExplainer``   for single LightGBM / XGBoost / CatBoost models
    - ``shap.KernelExplainer`` for calibrated ensembles (slower, model-agnostic)

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Must expose a ``predict_proba`` method.
    X_background : pd.DataFrame or np.ndarray
        Background dataset for KernelExplainer (sampled to ``sample_size`` rows).
    feature_names : list of str, optional
        Column names for SHAP value arrays. Inferred from X_background if DataFrame.
    sample_size : int
        Number of background samples for KernelExplainer.
    use_kernel : bool
        Force use of KernelExplainer even for tree models.
    """

    def __init__(
        self,
        model,
        X_background: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        sample_size:   int = SHAP_SAMPLE_SIZE,
        use_kernel:    bool = False,
    ) -> None:
        self.model         = model
        self.sample_size   = sample_size
        self.feature_names = (
            list(X_background.columns)
            if isinstance(X_background, pd.DataFrame)
            else feature_names
        )

        X_bg = self._sample(X_background)
        self.explainer_ = self._build_explainer(X_bg, use_kernel)
        logger.info("ShapExplainer initialised (%s).", type(self.explainer_).__name__)

    def _sample(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Down-sample background data for computational efficiency."""
        n = len(X)
        if n > self.sample_size:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(n, size=self.sample_size, replace=False)
            return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
        return X

    def _build_explainer(self, X_bg, use_kernel: bool):
        """Instantiate the most appropriate SHAP explainer."""
        if use_kernel:
            logger.info("Using KernelExplainer (model-agnostic, slower).")
            return shap.KernelExplainer(
                model=self.model.predict_proba,
                data=shap.sample(X_bg, nsamples=min(100, len(X_bg))),
                link="logit",
            )
        try:
            return shap.TreeExplainer(self.model)
        except Exception:
            logger.warning("TreeExplainer failed; falling back to KernelExplainer.")
            return shap.KernelExplainer(
                model=self.model.predict_proba,
                data=shap.sample(X_bg, nsamples=min(100, len(X_bg))),
                link="logit",
            )

    def compute_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        check_additivity: bool = False,
    ) -> np.ndarray:
        """
        Compute SHAP values for the input data.

        Parameters
        ----------
        X : array-like
        check_additivity : bool
            Passed to TreeExplainer (slower but validates SHAP sum = prediction).

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        shap_values = self.explainer_.shap_values(
            X,
            check_additivity=check_additivity,
        )
        # For binary classifiers, TreeExplainer returns a list → take class-1 values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        return np.array(shap_values)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def summary_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        plot_type:   str  = "dot",
        max_display: int  = SHAP_MAX_DISPLAY,
        save:        bool = True,
        filename:    str  = "shap_summary.png",
    ) -> None:
        """
        Global feature importance bee-swarm (or bar) plot.

        Parameters
        ----------
        X : array-like
        plot_type : str
            ``'dot'`` (bee-swarm) or ``'bar'`` (mean |SHAP|).
        max_display : int
            Maximum number of features to display.
        save : bool
            If True, save figure to ``PATHS.figures``.
        filename : str
            Output filename.
        """
        shap_values = self.compute_shap_values(X)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Summary — Global Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save:
            _save_figure(filename)
        plt.show()

    def dependence_plot(
        self,
        X:           Union[pd.DataFrame, np.ndarray],
        feature:     str,
        interaction: str  = "auto",
        save:        bool = True,
        filename:    Optional[str] = None,
    ) -> None:
        """
        SHAP Dependence Plot — shows feature's marginal effect and interactions.

        Parameters
        ----------
        X : array-like
        feature : str
            Name of the primary feature to plot.
        interaction : str
            Feature to use for colouring (interaction term). ``'auto'`` for automatic.
        save : bool
        filename : str, optional
        """
        shap_values = self.compute_shap_values(X)
        filename    = filename or f"shap_dependence_{feature}.png"
        shap.dependence_plot(
            feature, shap_values, X,
            feature_names=self.feature_names,
            interaction_index=interaction,
            show=False,
        )
        plt.title(f"SHAP Dependence — {feature}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            _save_figure(filename)
        plt.show()

    def waterfall_plot(
        self,
        X:       Union[pd.DataFrame, np.ndarray],
        row_idx: int  = 0,
        save:    bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """
        SHAP Waterfall Plot — local explanation for a single customer.

        Parameters
        ----------
        X : array-like
        row_idx : int
            Index of the customer to explain.
        save : bool
        filename : str, optional
        """
        filename    = filename or f"shap_waterfall_customer_{row_idx}.png"
        explanation = self.explainer_(X)
        if isinstance(explanation, list):
            explanation = explanation[1]

        shap.waterfall_plot(explanation[row_idx], show=False)
        plt.title(f"SHAP Waterfall — Customer #{row_idx}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            _save_figure(filename)
        plt.show()

    def bar_plot(
        self,
        X:           Union[pd.DataFrame, np.ndarray],
        max_display: int  = SHAP_MAX_DISPLAY,
        save:        bool = True,
        filename:    str  = "shap_bar.png",
    ) -> None:
        """
        Mean absolute SHAP value bar plot (global importance).
        """
        shap_values = self.compute_shap_values(X)
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title("Mean |SHAP| — Global Feature Importance", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            _save_figure(filename)
        plt.show()


# =============================================================================
# Helper Utilities
# =============================================================================
def _save_figure(filename: str, dpi: int = 200) -> Path:
    """Save the current matplotlib figure to PATHS.figures."""
    out_path = PATHS.figures / filename
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved → %s", out_path)
    return out_path


def plot_feature_importance_comparison(
    models:        dict,
    feature_names: List[str],
    top_n:         int  = 20,
    save:          bool = True,
    filename:      str  = "feature_importance_comparison.png",
) -> None:
    """
    Side-by-side bar chart comparing feature importances across multiple models.

    Parameters
    ----------
    models : dict
        Mapping of ``{model_name: fitted_model}``. Model must have
        a ``feature_importances_`` attribute.
    feature_names : list of str
    top_n : int
        Number of top features to display.
    save : bool
    filename : str
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        importances = pd.Series(
            model.feature_importances_, index=feature_names
        ).nlargest(top_n).sort_values()
        importances.plot(kind="barh", ax=ax, color="#2196F3", edgecolor="white")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance")

    plt.suptitle("Feature Importance Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        _save_figure(filename)
    plt.show()

