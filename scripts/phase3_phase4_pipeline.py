"""
Phase 3 & Phase 4 — End-to-End Execution Script
=================================================
Runs the complete Modeling, Calibration, SHAP Explainability & Decision Engine pipeline.
Generates:
  - reports/figures/05_model_benchmark_and_calibration.png
  - reports/figures/06_shap_summary_beeswarm.png
  - reports/figures/07_shap_dependence_tipping_points.png
  - reports/figures/08_profit_curve_and_2d_matrix.png
Prints the comprehensive FINAL EXECUTIVE REPORT.
"""

import os, sys, warnings, logging
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "matplotlib")

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, PATHS
from src.features import build_leakage_free_pipeline
from src.models import (
    Stage1EarlyWarningModel,
    tune_lgbm_with_optuna,
    ModelBenchmarkSuite
)
from src.explainability import ShapExplainer
from src.decision_engine import TelecomFinancialDecisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Phase3_4")


def run():
    print("=" * 80)
    print("  PHASE 3 & 4: END-TO-END MODELING, SHAP & QUANTITATIVE DECISION ENGINE")
    print("=" * 80)

    # 1. Load Data
    df = pd.read_parquet(ROOT / "data/processed/merged_clean.parquet")
    print(f"\n[DATA] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Targets
    y_churn = df["churn"].astype(int)
    EPS = 1e-5
    rev_decay_proxy = df["avg3rev"].fillna(0) / (df["avg6rev"].fillna(0) + EPS)
    # Define severe_compression: change_rev < -10.0 or rev_decay_3v6 < 0.85
    y_compression = (
        (df["change_rev"].fillna(0) < -10.0) | (rev_decay_proxy < 0.85)
    ).astype(int)

    features_raw = df.drop(columns=["Customer_ID", "churn"], errors="ignore")

    # Stratified 80/20 Split
    X_train_raw, X_test_raw, y_train, y_test, y_comp_train, y_comp_test = train_test_split(
        features_raw, y_churn, y_compression,
        test_size=0.20, stratify=y_churn, random_state=RANDOM_STATE
    )
    print(f"[SPLIT] Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")
    print(f"[PREVALENCE] Churn: Train = {y_train.mean():.2%}, Test = {y_test.mean():.2%}")
    print(f"[PREVALENCE] Revenue Compression: Train = {y_comp_train.mean():.2%}, Test = {y_comp_test.mean():.2%}")

    # 2. Run Feature Engineering Pipeline
    print("\n" + "-" * 80)
    print("  STEP 1: RUNNING LEAKAGE-FREE FEATURE ENGINEERING PIPELINE")
    print("-" * 80)
    pipeline, oof_enc = build_leakage_free_pipeline(flag_threshold=0.10)

    X_tr_s12 = pipeline.fit_transform(X_train_raw)
    X_te_s12 = pipeline.transform(X_test_raw)

    X_train_fe = oof_enc.fit_transform(X_tr_s12, y_train)
    X_test_fe  = oof_enc.transform(X_te_s12)

    # Ensure numeric matrix
    X_train_num = X_train_fe.select_dtypes("number")
    X_test_num  = X_test_fe.select_dtypes("number")
    print(f"[FEATURES READY] Train Shape: {X_train_num.shape} | Test Shape: {X_test_num.shape}")

    # 3. Stage 1 — Early Warning Model (Revenue Compression)
    print("\n" + "-" * 80)
    print("  STEP 2: TRAINING STAGE 1 EARLY WARNING MODEL (REVENUE COMPRESSION)")
    print("-" * 80)
    stage1_model = Stage1EarlyWarningModel(random_state=RANDOM_STATE)
    stage1_model.fit(X_train_num, y_comp_train)
    stage1_metrics = stage1_model.evaluate(X_test_num, y_comp_test)

    p_comp_test = stage1_model.predict_proba(X_test_num)

    print("  Stage 1 Evaluation Metrics:")
    for k, v in stage1_metrics.items():
        print(f"    {k:<15}: {v:.4f}")

    # 4. Stage 2 — Optuna Tuning & Model Benchmark
    print("\n" + "-" * 80)
    print("  STEP 3: OPTUNA BAYESIAN TUNING FOR LIGHTGBM")
    print("-" * 80)
    best_lgb_params, best_lgb_score = tune_lgbm_with_optuna(
        X_train_num, y_train, n_trials=25, timeout=120, random_state=RANDOM_STATE
    )
    print(f"  Optuna Completed! Best Val ROC-AUC: {best_lgb_score:.4f}")
    print(f"  Best Parameters: {best_lgb_params}")

    print("\n" + "-" * 80)
    print("  STEP 4: TRAINING BENCHMARK SUITE & ENSEMBLE CALIBRATION")
    print("-" * 80)
    benchmark_suite = ModelBenchmarkSuite(random_state=RANDOM_STATE)
    benchmark_df = benchmark_suite.fit_and_benchmark(
        X_train_num, y_train, X_test_num, y_test, lgb_best_params=best_lgb_params
    )

    print("\n" + "=" * 80)
    print("  STAGE 2 MODEL BENCHMARK TABLE")
    print("=" * 80)
    print(benchmark_df.to_string(index=False))

    # Plot Figure 05: Benchmark & Calibration
    fig05_path = str(PATHS.figures / "05_model_benchmark_and_calibration.png")
    benchmark_suite.plot_benchmark_and_calibration(X_test_num, y_test, save_path=fig05_path)
    print(f"\n  [SAVED] {fig05_path}")

    # Get calibrated churn probabilities
    calibrated_model = benchmark_suite.calibrated_ensemble
    p_churn_test = calibrated_model.predict_proba(X_test_num)[:, 1]

    # 5. SHAP Explainability
    print("\n" + "-" * 80)
    print("  STEP 5: COMPUTING SHAP VALUES & NON-LINEAR TIPPING POINTS")
    print("-" * 80)
    best_lgb_model = benchmark_suite.models["LightGBM (Tuned)"]
    explainer = ShapExplainer(model=best_lgb_model, feature_names=X_test_num.columns.tolist())
    explainer.compute_shap(X_test_num, sample_size=5000)

    fig06_path = str(PATHS.figures / "06_shap_summary_beeswarm.png")
    explainer.plot_summary_beeswarm(max_display=20, save_path=fig06_path)
    print(f"  [SAVED] {fig06_path}")

    fig07_path = str(PATHS.figures / "07_shap_dependence_tipping_points.png")
    explainer.plot_dependence_tipping_points(
        primary_feature_1="eqpdays", interaction_feature_1="hnd_price",
        primary_feature_2="rev_per_mou", interaction_feature_2="totmrc_Mean",
        save_path=fig07_path
    )
    print(f"  [SAVED] {fig07_path}")

    # 6. Financial Decision Engine & Strategy Benchmark
    print("\n" + "-" * 80)
    print("  STEP 6: FINANCIAL DECISION ENGINE & 2D STRATEGY MATRIX")
    print("-" * 80)
    rev_mean_test = X_test_raw["rev_Mean"].fillna(50.0).values
    eqpdays_test  = X_test_raw["eqpdays"].fillna(340.0).values

    decision_engine = TelecomFinancialDecisionEngine(
        p_churn=p_churn_test,
        p_comp=p_comp_test,
        rev_mean=rev_mean_test,
        eqpdays=eqpdays_test,
        scale_factor=5.0
    )

    optimal_p, profit_curve_df = decision_engine.optimize_threshold(p_comp_threshold=0.35)
    print(f"\n  Economic Optimum Found: Threshold p* = {optimal_p:.2f}")

    strategy_benchmark = decision_engine.benchmark_strategies(optimal_p=optimal_p, p_comp_threshold=0.35)

    print("\n" + "=" * 80)
    print("  EXECUTIVE STRATEGY BENCHMARK (SCALED TO 100,000 CUSTOMER POPULATION)")
    print("=" * 80)
    print(strategy_benchmark.to_string(index=False))

    # Quadrant breakdown at optimal threshold
    df_quadrants = decision_engine.assign_quadrants(optimal_p, p_comp_threshold=0.35)
    quad_counts = df_quadrants["quadrant"].value_counts()
    print("\n  Customer Distribution Across 2D Strategic Quadrants (Test Set / Scaled):")
    for q_name, cnt in quad_counts.items():
        print(f"    {q_name:<24}: {cnt:5,d} test ({cnt*5:6,d} scaled, {cnt/len(df_quadrants)*100:5.1f}%)")

    fig08_path = str(PATHS.figures / "08_profit_curve_and_2d_matrix.png")
    decision_engine.plot_profit_curve_and_2d_matrix(
        optimal_p=optimal_p, curve_df=profit_curve_df, p_comp_threshold=0.35, save_path=fig08_path
    )
    print(f"\n  [SAVED] {fig08_path}")

    # 7. Final Executive Summary
    print("\n" + "=" * 80)
    print("  FINAL EXECUTIVE SUMMARY — PHASES 3 & 4 COMPLETE")
    print("=" * 80)
    print(f"  1. Early Warning Model (Silent Attrition) ROC-AUC : {stage1_metrics['ROC-AUC']:.4f}")
    print(f"  2. Stage 2 Best Calibrated Ensemble ROC-AUC       : {benchmark_df.loc[0, 'ROC-AUC']:.4f}")
    print(f"  3. Brier Score Calibration (Raw -> Calibrated)    : {benchmark_df[benchmark_df['Model'].str.contains('Calibrated')]['Brier Score'].values[0]:.4f}")
    print(f"  4. Optimal Decision Cutoff (Threshold*)          : p* = {optimal_p:.2f}")
    opt_net = strategy_benchmark.loc[3, "Net Profit Lift ($M)"]
    opt_roi = strategy_benchmark.loc[3, "ROI (%)"]
    opt_cost = strategy_benchmark.loc[3, "Campaign Cost ($M)"]
    opt_saved = strategy_benchmark.loc[3, "Saved Revenue ($M)"]
    print(f"  5. Targeted Campaign Cost                         : ${opt_cost:.2f}M USD")
    print(f"  6. Gross Preserved LTV Revenue                    : ${opt_saved:.2f}M USD")
    print(f"  7. Net Economic Profit Lift vs Status Quo         : ${opt_net:.2f}M USD")
    print(f"  8. Return on Retention Investment (ROI)           : {opt_roi:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    run()

