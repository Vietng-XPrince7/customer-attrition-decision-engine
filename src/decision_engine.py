"""
decision_engine.py — Quantitative Financial Decision Engine & Profit Optimization
==================================================================================
Translates raw model risk probabilities (Churn and Revenue Compression) into
profit-maximizing executive decisions using:

1. **Telecom Financial Parameters**:
   - Customer Lifetime Value (LTV): LTV_i = rev_Mean_i * 12 * Gross_Margin (70%)
   - Asymmetric cost-benefit matrix across 4 targeted quadrants.

2. **2D Risk-Value Matrix (VIP & Cohort Level)**:
   - Q1: Silent Eroders     (P_churn < p*, P_comp >= p_comp*) -> Right-Sizing ($15, R=65%)
   - Q2: Critical Bleeders  (P_churn >= p*, P_comp >= p_comp* or eqpdays > 540) -> Device Subsidy ($120, R=55%)
   - Q3: Sudden Churners    (P_churn >= p*, Low Comp) -> Proactive Care / Waiver ($35, R=50%)
   - Q4: Safe Customers     (P_churn < p*, Low Comp) -> Organic Engagement ($0)

3. **Expected Profit Curve & Optimal Threshold Search (Threshold*)**:
   - Sweeps thresholds p in [0.05, 0.95] to find the economic optimum.
   - Benchmarks 4 strategies: Do Nothing, Blanket, Default ML (0.50), and 2D Matrix Optimal.

4. **Visualization**:
   - reports/figures/08_profit_curve_and_2d_matrix.png (300 DPI)
"""

import os, logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import PATHS, FINANCIAL, RANDOM_STATE

logger = logging.getLogger(__name__)


class TelecomFinancialDecisionEngine:
    """
    Financial Decision Engine implementing Expected Value Maximization
    under the 2D Risk-Value Matrix framework.
    """

    GROSS_MARGIN: float = 0.70       # 70% Gross Margin
    ANNUAL_MONTHS: int = 12

    # Intervention Cost & Success Rates per Quadrant
    STRATEGY_PARAMS = {
        "Q1_Silent_Eroders":    {"cost": 15.0,  "success_rate": 0.65, "name": "Plan Right-Sizing & Bonus Mins"},
        "Q2_Critical_Bleeders": {"cost": 120.0, "success_rate": 0.55, "name": "VIP Device Subsidy & 12M Lock-in"},
        "Q3_Sudden_Churners":   {"cost": 35.0,  "success_rate": 0.50, "name": "Proactive Care & Overage Waiver"},
        "Q4_Safe_Base":         {"cost": 0.0,   "success_rate": 0.00, "name": "Organic Engagement (Zero Spend)"},
    }

    def __init__(
        self,
        p_churn: np.ndarray,
        p_comp: np.ndarray,
        rev_mean: np.ndarray,
        eqpdays: Optional[np.ndarray] = None,
        scale_factor: float = 5.0  # Scale 20,000 test set to 100,000 annual population
    ) -> None:
        self.p_churn = np.asarray(p_churn, dtype=float)
        self.p_comp = np.asarray(p_comp, dtype=float)
        self.rev_mean = np.asarray(rev_mean, dtype=float)
        self.eqpdays = np.asarray(eqpdays if eqpdays is not None else np.zeros_like(p_churn), dtype=float)
        self.scale_factor = scale_factor

        # Annual Gross Profit LTV per customer
        self.ltv = np.maximum(self.rev_mean, 5.0) * self.ANNUAL_MONTHS * self.GROSS_MARGIN

    def assign_quadrants(
        self,
        p_churn_threshold: float,
        p_comp_threshold: float = 0.35,
        eqp_tipping_point: float = 540.0
    ) -> pd.DataFrame:
        """
        Classify each customer into one of 4 Strategic Quadrants.
        """
        n = len(self.p_churn)
        quadrants = np.empty(n, dtype=object)

        is_high_churn = self.p_churn >= p_churn_threshold
        is_high_comp = (self.p_comp >= p_comp_threshold) | (self.eqpdays > eqp_tipping_point)

        # Vectorized assignment
        q2_mask = is_high_churn & is_high_comp
        q1_mask = (~is_high_churn) & is_high_comp
        q3_mask = is_high_churn & (~is_high_comp)
        q4_mask = (~is_high_churn) & (~is_high_comp)

        quadrants[q1_mask] = "Q1_Silent_Eroders"
        quadrants[q2_mask] = "Q2_Critical_Bleeders"
        quadrants[q3_mask] = "Q3_Sudden_Churners"
        quadrants[q4_mask] = "Q4_Safe_Base"

        df_out = pd.DataFrame({
            "p_churn": self.p_churn,
            "p_comp": self.p_comp,
            "rev_mean": self.rev_mean,
            "eqpdays": self.eqpdays,
            "ltv": self.ltv,
            "quadrant": quadrants
        })
        return df_out

    def compute_expected_net_profit(
        self,
        p_churn_threshold: float,
        p_comp_threshold: float = 0.35
    ) -> Dict[str, float]:
        """
        Calculate the Expected Net Financial Benefit of the strategy at a given threshold.
        """
        df = self.assign_quadrants(p_churn_threshold, p_comp_threshold)

        total_cost = 0.0
        saved_ltv = 0.0

        for q_name, params in self.STRATEGY_PARAMS.items():
            mask = df["quadrant"] == q_name
            n_cust = mask.sum()
            if n_cust == 0:
                continue

            q_ltv = df.loc[mask, "ltv"].values
            q_p_churn = df.loc[mask, "p_churn"].values

            cost_per_cust = params["cost"]
            success_rate = params["success_rate"]

            if q_name in ["Q2_Critical_Bleeders", "Q3_Sudden_Churners"]:
                # Churn prevention intervention
                cost = n_cust * cost_per_cust
                # Expected saved LTV: probability customer was going to churn * success rate * LTV
                benefit = np.sum(q_p_churn * success_rate * q_ltv)
            elif q_name == "Q1_Silent_Eroders":
                # Revenue compression defense: protects 30% of LTV erosion
                cost = n_cust * cost_per_cust
                benefit = np.sum(0.30 * success_rate * q_ltv)
            else:
                cost = 0.0
                benefit = 0.0

            total_cost += cost
            saved_ltv += benefit

        net_profit = saved_ltv - total_cost

        return {
            "threshold": p_churn_threshold,
            "cost_test": total_cost,
            "saved_ltv_test": saved_ltv,
            "net_profit_test": net_profit,
            "cost_scaled": total_cost * self.scale_factor,
            "saved_ltv_scaled": saved_ltv * self.scale_factor,
            "net_profit_scaled": net_profit * self.scale_factor,
            "roi_pct": (net_profit / total_cost * 100) if total_cost > 0 else 0.0,
        }

    def optimize_threshold(
        self,
        p_comp_threshold: float = 0.35,
        n_steps: int = 91
    ) -> Tuple[float, pd.DataFrame]:
        """
        Sweep thresholds in [0.05, 0.95] to find the profit-maximizing threshold p*.
        """
        thresholds = np.linspace(0.05, 0.95, n_steps)
        records = []
        for t in thresholds:
            res = self.compute_expected_net_profit(t, p_comp_threshold)
            records.append(res)

        curve_df = pd.DataFrame(records)
        best_idx = curve_df["net_profit_scaled"].idxmax()
        optimal_threshold = float(curve_df.loc[best_idx, "threshold"])

        logger.info("Optimal threshold p* = %.4f with Net Profit Lift = $%.2fM",
                    optimal_threshold, curve_df.loc[best_idx, "net_profit_scaled"] / 1e6)

        return optimal_threshold, curve_df

    def benchmark_strategies(
        self,
        optimal_p: float,
        p_comp_threshold: float = 0.35
    ) -> pd.DataFrame:
        """
        Compare the 4 executive strategies scaled to 100,000 customers.
        """
        # (a) Do Nothing
        do_nothing = {
            "Strategy": "1. Do Nothing (Status Quo)",
            "Targeted Customers": 0,
            "Campaign Cost ($M)": 0.0,
            "Saved Revenue ($M)": 0.0,
            "Net Profit Lift ($M)": 0.0,
            "ROI (%)": 0.0
        }

        # (b) Blanket Campaign (Treat All at low-cost generic offer $35)
        n_all = len(self.p_churn) * self.scale_factor
        blanket_cost = n_all * 35.0 / 1e6
        blanket_saved = np.sum(self.p_churn * 0.25 * self.ltv) * self.scale_factor / 1e6
        blanket_net = blanket_saved - blanket_cost
        blanket = {
            "Strategy": "2. Blanket Campaign (Treat All)",
            "Targeted Customers": int(n_all),
            "Campaign Cost ($M)": blanket_cost,
            "Saved Revenue ($M)": blanket_saved,
            "Net Profit Lift ($M)": blanket_net,
            "ROI (%)": (blanket_net / blanket_cost * 100) if blanket_cost > 0 else 0.0
        }

        # (c) Default ML Threshold (p = 0.50)
        p50_res = self.compute_expected_net_profit(0.50, p_comp_threshold)
        df_p50 = self.assign_quadrants(0.50, p_comp_threshold)
        n_p50 = (df_p50["quadrant"] != "Q4_Safe_Base").sum() * self.scale_factor
        p50 = {
            "Strategy": "3. Default ML (Threshold = 0.50)",
            "Targeted Customers": int(n_p50),
            "Campaign Cost ($M)": p50_res["cost_scaled"] / 1e6,
            "Saved Revenue ($M)": p50_res["saved_ltv_scaled"] / 1e6,
            "Net Profit Lift ($M)": p50_res["net_profit_scaled"] / 1e6,
            "ROI (%)": p50_res["roi_pct"]
        }

        # (d) Economic Optimum (p = optimal_p) + 2D Matrix
        opt_res = self.compute_expected_net_profit(optimal_p, p_comp_threshold)
        df_opt = self.assign_quadrants(optimal_p, p_comp_threshold)
        n_opt = (df_opt["quadrant"] != "Q4_Safe_Base").sum() * self.scale_factor
        opt = {
            "Strategy": f"4. 2D Risk-Value Engine (Optimal p*={optimal_p:.2f})",
            "Targeted Customers": int(n_opt),
            "Campaign Cost ($M)": opt_res["cost_scaled"] / 1e6,
            "Saved Revenue ($M)": opt_res["saved_ltv_scaled"] / 1e6,
            "Net Profit Lift ($M)": opt_res["net_profit_scaled"] / 1e6,
            "ROI (%)": opt_res["roi_pct"]
        }

        return pd.DataFrame([do_nothing, blanket, p50, opt])

    def plot_profit_curve_and_2d_matrix(
        self,
        optimal_p: float,
        curve_df: pd.DataFrame,
        p_comp_threshold: float = 0.35,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Generate publication plot (reports/figures/08_profit_curve_and_2d_matrix.png).
        """
        df_seg = self.assign_quadrants(optimal_p, p_comp_threshold)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Panel 1: Expected Profit Curve
        ax = axes[0]
        ax.plot(
            curve_df["threshold"], curve_df["net_profit_scaled"] / 1e6,
            color="#0D47A1", lw=3, label="Expected Net Profit Lift ($M)"
        )
        ax.plot(
            curve_df["threshold"], curve_df["saved_ltv_scaled"] / 1e6,
            color="#2E7D32", lw=2, linestyle="--", label="Gross Saved LTV ($M)"
        )
        ax.plot(
            curve_df["threshold"], curve_df["cost_scaled"] / 1e6,
            color="#C62828", lw=2, linestyle=":", label="Campaign Cost ($M)"
        )

        max_profit = curve_df["net_profit_scaled"].max() / 1e6
        ax.axvline(optimal_p, color="#D50000", linestyle="--", lw=2,
                   label=f"Optimal Threshold p* = {optimal_p:.2f}")
        ax.scatter([optimal_p], [max_profit], color="#D50000", s=120, zorder=5)
        ax.annotate(
            f"Maximum Net Profit Lift:\n${max_profit:.2f}M USD",
            xy=(optimal_p, max_profit),
            xytext=(optimal_p + 0.08, max_profit - 0.5),
            fontsize=10, fontweight="bold", color="#D50000",
            arrowprops=dict(arrowstyle="->", color="#D50000", lw=2)
        )

        ax.axhline(0, color="black", linestyle="-", lw=1, alpha=0.5)
        ax.set_title("Expected Net Profit Curve vs. Decision Threshold (100k Population)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Decision Threshold (p)", fontsize=11)
        ax.set_ylabel("USD Millions ($M)", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=10)

        # Panel 2: 2D Risk-Value Strategic Matrix Scatter
        ax2 = axes[1]
        colors = {
            "Q1_Silent_Eroders":    "#FF9800",  # Orange
            "Q2_Critical_Bleeders": "#E53935",  # Red
            "Q3_Sudden_Churners":   "#9C27B0",  # Purple
            "Q4_Safe_Base":         "#43A047",  # Green
        }
        labels = {
            "Q1_Silent_Eroders":    "Q1: Silent Eroders (Right-Sizing)",
            "Q2_Critical_Bleeders": "Q2: Critical Bleeders (Device Subsidy)",
            "Q3_Sudden_Churners":   "Q3: Sudden Churners (Care & Waiver)",
            "Q4_Safe_Base":         "Q4: Safe Base (Zero Spend)",
        }

        # Subsample 3,000 points for clear visualization
        df_sub = df_seg.sample(n=min(3000, len(df_seg)), random_state=RANDOM_STATE)

        for q_key, color in colors.items():
            sub = df_sub[df_sub["quadrant"] == q_key]
            ax2.scatter(
                sub["p_churn"], sub["p_comp"],
                c=color, alpha=0.5, s=25, label=labels[q_key], edgecolors="none"
            )

        ax2.axvline(optimal_p, color="black", linestyle="--", lw=1.5, label=f"Churn Cutoff p* = {optimal_p:.2f}")
        ax2.axhline(p_comp_threshold, color="black", linestyle=":", lw=1.5, label=f"Comp Cutoff = {p_comp_threshold:.2f}")

        ax2.set_title("2D Risk-Value Matrix: Dynamic Customer Segmentation", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Calibrated Churn Probability P(Churn)", fontsize=11)
        ax2.set_ylabel("Revenue Compression Score P(Compression)", fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

        fig.suptitle("Financial Decision Engine — Economic Optimization & Action Matrix", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("Saved profit curve and 2D matrix plot to %s", save_path)
        plt.close()
