"""
decision_engine.py — Financial Decision Engine
================================================
Translates model probability scores into business-optimal decisions using:

1. **2D Risk-Value Matrix**
   Segments customers into four quadrants: High/Low Churn Risk × High/Low CLV.
   Each quadrant maps to a distinct retention strategy.

2. **Expected Profit Curve**
   Sweeps probability thresholds [0, 1] and computes expected profit at each
   threshold using the asymmetric profit matrix from ``src.config.FINANCIAL``.

3. **Optimal Threshold (Threshold\*)**
   The cut-off that maximises cumulative expected profit, accounting for
   intervention costs and false-alarm penalties.

Usage
-----
    from src.decision_engine import DecisionEngine

    engine = DecisionEngine(proba=y_pred_proba, clv=df["Balance"])
    engine.plot_risk_value_matrix()
    engine.plot_expected_profit_curve()
    threshold_star = engine.find_optimal_threshold()
    segments = engine.segment_customers(threshold=threshold_star)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.config import FINANCIAL, PATHS, RANDOM_STATE

logger = logging.getLogger(__name__)


# =============================================================================
# Decision Engine
# =============================================================================
class DecisionEngine:
    """
    Financial Decision Engine for churn-driven revenue optimisation.

    Parameters
    ----------
    proba : array-like of shape (n_samples,)
        Calibrated churn probabilities from Stage-2 ensemble.
    clv : array-like of shape (n_samples,), optional
        Customer Lifetime Value estimates. Defaults to ``FINANCIAL.CLV_mean``.
    customer_ids : array-like, optional
        Customer identifiers for the output segment table.
    financial : _Financial, optional
        Financial parameters. Defaults to ``FINANCIAL`` from config.
    """

    def __init__(
        self,
        proba:         np.ndarray,
        clv:           Optional[np.ndarray] = None,
        customer_ids:  Optional[np.ndarray] = None,
        financial=None,
    ) -> None:
        self.proba        = np.asarray(proba, dtype=float)
        self.clv          = (
            np.asarray(clv, dtype=float)
            if clv is not None
            else np.full(len(self.proba), FINANCIAL.CLV_mean)
        )
        self.customer_ids = customer_ids
        self.financial    = financial or FINANCIAL

        assert len(self.proba) == len(self.clv), "proba and clv must have the same length."

    # ── Expected Profit ────────────────────────────────────────────────────────

    def expected_profit_at_threshold(self, threshold: float) -> float:
        """
        Compute total expected profit when the decision threshold is ``threshold``.

        Profit matrix (per customer)
        ----------------------------
        - Predicted churner (p ≥ threshold) & actually churns → TP_benefit
        - Predicted churner (p ≥ threshold) & stays          → FP_cost
        - Predicted stayer  (p <  threshold) & actually churns→ FN_cost (missed)
        - Predicted stayer  (p <  threshold) & stays          → TN_benefit

        Note: Because y_true is not available at inference time, we compute
        *expected* profit using ``p`` as the probability of churn.

        ``E[profit | customer_i, threshold] =``
        ``  p_i × TP + (1 - p_i) × FP   if predicted = churner``
        ``  p_i × FN + (1 - p_i) × TN   if predicted = stayer``

        Parameters
        ----------
        threshold : float

        Returns
        -------
        float
            Sum of expected profit across all customers.
        """
        p       = self.proba
        fin     = self.financial
        treated = p >= threshold

        profit_treated = treated   * (p * fin.TP_benefit + (1 - p) * fin.FP_cost)
        profit_untreated = ~treated * (p * fin.FN_cost   + (1 - p) * fin.TN_benefit)

        total = float((profit_treated + profit_untreated).sum())
        return total

    def expected_profit_curve(
        self,
        n_thresholds: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep thresholds from 0 to 1 and compute expected profit at each point.

        Parameters
        ----------
        n_thresholds : int
            Number of threshold values to evaluate.

        Returns
        -------
        thresholds : np.ndarray
        profits : np.ndarray
        """
        thresholds = np.linspace(0.0, 1.0, n_thresholds)
        profits = np.array([self.expected_profit_at_threshold(t) for t in thresholds])
        return thresholds, profits

    def find_optimal_threshold(self, n_thresholds: int = 200) -> float:
        """
        Find the probability threshold that maximises expected profit.

        Returns
        -------
        float
            Threshold* — the optimal decision cut-off.
        """
        thresholds, profits = self.expected_profit_curve(n_thresholds)
        idx = int(np.argmax(profits))
        threshold_star = float(thresholds[idx])
        max_profit     = float(profits[idx])
        logger.info(
            "Threshold* = %.4f → Expected Profit = $%.2f", threshold_star, max_profit
        )
        return threshold_star

    # ── Segmentation ───────────────────────────────────────────────────────────

    def segment_customers(
        self,
        threshold:  float,
        clv_median: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Classify customers into the 2D Risk-Value matrix quadrants.

        Quadrants
        ---------
        - **Q1 – Champions**     : Low churn risk, High CLV → Nurture & upsell
        - **Q2 – Priority Save** : High churn risk, High CLV → Aggressive retention
        - **Q3 – Watch List**    : High churn risk, Low CLV  → Cost-efficient outreach
        - **Q4 – Steady Base**   : Low churn risk, Low CLV   → Maintain engagement

        Parameters
        ----------
        threshold : float
            Decision threshold (use ``find_optimal_threshold()``).
        clv_median : float, optional
            CLV split point. Defaults to sample median.

        Returns
        -------
        pd.DataFrame
            One row per customer with columns:
            ['customer_id', 'churn_proba', 'clv', 'churn_flag', 'clv_tier', 'segment']
        """
        clv_median = clv_median or float(np.median(self.clv))
        churn_flag = (self.proba >= threshold).astype(int)
        clv_tier   = (self.clv >= clv_median).astype(int)

        def _assign_segment(cf: int, ct: int) -> str:
            if cf == 0 and ct == 1: return "Q1 – Champions"
            if cf == 1 and ct == 1: return "Q2 – Priority Save"
            if cf == 1 and ct == 0: return "Q3 – Watch List"
            return "Q4 – Steady Base"

        segments = [_assign_segment(cf, ct) for cf, ct in zip(churn_flag, clv_tier)]

        df = pd.DataFrame({
            "customer_id": self.customer_ids if self.customer_ids is not None else np.arange(len(self.proba)),
            "churn_proba": self.proba,
            "clv":         self.clv,
            "churn_flag":  churn_flag,
            "clv_tier":    clv_tier,
            "segment":     segments,
        })
        logger.info("Segmentation summary:\n%s", df["segment"].value_counts().to_string())
        return df

    # ── Visualisations ─────────────────────────────────────────────────────────

    def plot_expected_profit_curve(
        self,
        n_thresholds: int  = 200,
        save:         bool = True,
        filename:     str  = "expected_profit_curve.png",
    ) -> None:
        """
        Plot the Expected Profit Curve and annotate Threshold*.

        Parameters
        ----------
        n_thresholds : int
        save : bool
        filename : str
        """
        thresholds, profits = self.expected_profit_curve(n_thresholds)
        idx            = int(np.argmax(profits))
        threshold_star = thresholds[idx]
        max_profit     = profits[idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(thresholds, profits / 1e6, color="#1565C0", linewidth=2.5, label="Expected Profit")
        ax.axvline(threshold_star, color="#D32F2F", linestyle="--", linewidth=2,
                   label=f"Threshold* = {threshold_star:.3f}")
        ax.axhline(max_profit / 1e6, color="#388E3C", linestyle=":", linewidth=1.5,
                   label=f"Max Profit = \${max_profit / 1e6:.2f}M")
        ax.scatter([threshold_star], [max_profit / 1e6], color="#D32F2F", zorder=5, s=80)
        ax.set_xlabel("Probability Threshold", fontsize=12)
        ax.set_ylabel("Expected Profit (USD Millions)", fontsize=12)
        ax.set_title("Expected Profit Curve — Optimal Decision Threshold", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            _save_figure(filename)
        plt.show()

    def plot_risk_value_matrix(
        self,
        threshold:  float,
        clv_median: Optional[float] = None,
        save:       bool = True,
        filename:   str  = "risk_value_matrix.png",
    ) -> None:
        """
        2D Scatter plot of Churn Probability vs. CLV, coloured by segment quadrant.

        Parameters
        ----------
        threshold : float
            Decision threshold.
        clv_median : float, optional
        save : bool
        filename : str
        """
        df = self.segment_customers(threshold, clv_median)
        clv_median = clv_median or float(np.median(self.clv))

        palette = {
            "Q1 – Champions":    "#43A047",
            "Q2 – Priority Save":"#E53935",
            "Q3 – Watch List":   "#FB8C00",
            "Q4 – Steady Base":  "#1E88E5",
        }

        fig, ax = plt.subplots(figsize=(10, 7))
        for seg, grp in df.groupby("segment"):
            ax.scatter(
                grp["churn_proba"], grp["clv"],
                label=seg, alpha=0.5, s=20,
                color=palette.get(seg, "grey"),
            )
        ax.axvline(threshold,  color="black", linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"Threshold* = {threshold:.3f}")
        ax.axhline(clv_median, color="black", linestyle=":",  linewidth=1.5, alpha=0.7,
                   label=f"CLV Median = \${clv_median:,.0f}")
        ax.set_xlabel("Churn Probability (Calibrated)", fontsize=12)
        ax.set_ylabel("Customer Lifetime Value (USD)",  fontsize=12)
        ax.set_title("2D Risk-Value Matrix — Customer Segmentation", fontsize=14, fontweight="bold")
        ax.legend(markerscale=2, fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            _save_figure(filename)
        plt.show()

    def segment_summary(self, threshold: float) -> pd.DataFrame:
        """
        Tabular summary of segment sizes and aggregate CLV.

        Parameters
        ----------
        threshold : float

        Returns
        -------
        pd.DataFrame
            Columns: segment, n_customers, pct, mean_clv, total_clv_at_risk
        """
        df = self.segment_customers(threshold)
        summary = (
            df.groupby("segment")
            .agg(
                n_customers  = ("customer_id", "count"),
                mean_churn   = ("churn_proba", "mean"),
                mean_clv     = ("clv",         "mean"),
                total_clv    = ("clv",         "sum"),
            )
            .reset_index()
        )
        summary["pct"] = summary["n_customers"] / summary["n_customers"].sum() * 100
        return summary.sort_values("total_clv", ascending=False)


# =============================================================================
# Helper
# =============================================================================
def _save_figure(filename: str, dpi: int = 200) -> None:
    out_path = PATHS.figures / filename
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved → %s", out_path)

