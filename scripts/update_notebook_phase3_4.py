"""
Script to update notebooks/OmnicampusAccountName.ipynb with Phase 3 & 4 implementation.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "OmnicampusAccountName.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update imports in Cell 2
nb["cells"][2]["source"] = [
    "# ── Standard imports ──────────────────────────────────────────────────────────\n",
    "import warnings\n",
    "import logging\n",
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from IPython.display import display, Image\n",
    "\n",
    "# ── Add src to path (for standalone notebook execution) ───────────────────────\n",
    "sys.path.insert(0, str(Path.cwd().parent))\n",
    "\n",
    "from src.config       import PATHS, COLUMNS, FINANCIAL, RANDOM_STATE\n",
    "from src.data_loader  import load_and_merge, save_processed, get_summary_stats\n",
    "from src.features     import (\n",
    "    DomainFeatureGenerator, MissingnessAndOutlierHandler,\n",
    "    OutOfFoldTargetEncoder, build_leakage_free_pipeline,\n",
    "    validate_pipeline,\n",
    ")\n",
    "from src.models import (\n",
    "    Stage1EarlyWarningModel, tune_lgbm_with_optuna,\n",
    "    ModelBenchmarkSuite\n",
    ")\n",
    "from src.explainability  import ShapExplainer\n",
    "from src.decision_engine import TelecomFinancialDecisionEngine\n",
    "\n",
    "# ── Global settings ───────────────────────────────────────────────────────────\n",
    "warnings.filterwarnings('ignore')\n",
    "logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')\n",
    "plt.rcParams.update({\n",
    "    'figure.dpi': 120,\n",
    "    'font.family': 'DejaVu Sans',\n",
    "    'axes.spines.top': False,\n",
    "    'axes.spines.right': False,\n",
    "})\n",
    "np.random.seed(RANDOM_STATE)\n",
    "print(f'RANDOM_STATE = {RANDOM_STATE}')\n"
]

# Replacement cells for Section 6, 7, 8
new_phase3_4_cells = [
    {
        "cell_type": "markdown",
        "id": "cell-s6-header",
        "metadata": {},
        "source": [
            "---\n",
            "## 6. Model Training, Optuna Hyperparameter Tuning & Probability Calibration\n",
            "\n",
            "### Architectural Principles\n",
            "1. **Stage 1 (Early Warning: Silent Attrition)**: Predicts `severe_compression` without target leakage (strictly omitting direct revenue-delta variables).\n",
            "2. **Stage 2 (Benchmark Suite & Stacking Ensemble)**: Benchmarks 5 individual models (`LogisticRegression`, `RandomForest`, `XGBoost`, `CatBoost`, `LightGBM Tuned`), builds a **Soft Voting Ensemble**, and applies **Isotonic Probability Calibration** via `CalibratedClassifierCV`.\n",
            "3. **Decision-Grade Probability Calibration**: Brier Score minimization ensures $P(\\text{churn})$ accurately reflects empirical departure frequencies."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-s6-stage1-run",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Stage 1: Early Warning Model (Revenue Compression) ────────────────────────\n",
            "stage1_model = Stage1EarlyWarningModel(random_state=RANDOM_STATE)\n",
            "stage1_model.fit(X_train_final, y_rev_train)\n",
            "stage1_metrics = stage1_model.evaluate(X_test_final, y_rev_test)\n",
            "\n",
            "print('=' * 65)\n",
            "print('  STAGE 1 EARLY WARNING MODEL EVALUATION (REVENUE COMPRESSION)')\n",
            "print('=' * 65)\n",
            "for k, v in stage1_metrics.items():\n",
            "    print(f'  {k:<18}: {v:.4f}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-s6-stage2-benchmark",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Stage 2: Model Benchmark Suite & Calibration ──────────────────────────────\n",
            "benchmark_suite = ModelBenchmarkSuite(random_state=RANDOM_STATE)\n",
            "benchmark_df = benchmark_suite.fit_and_benchmark(\n",
            "    X_train_final, y_train, X_test_final, y_test\n",
            ")\n",
            "\n",
            "print('=' * 80)\n",
            "print('  STAGE 2 MODEL BENCHMARK & CALIBRATION TABLE')\n",
            "print('=' * 80)\n",
            "display(benchmark_df)\n",
            "\n",
            "# Display Diagnostic Plot (ROC, PR, Calibration, Probability Distribution)\n",
            "display(Image(filename=str(PATHS.figures / '05_model_benchmark_and_calibration.png')))\n"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "cell-s7-header",
        "metadata": {},
        "source": [
            "---\n",
            "## 7. Explainable AI (SHAP) & Non-Linear Tipping Points Discovery\n",
            "\n",
            "Using Game-Theoretic Shapley values (`shap.TreeExplainer`) on our best Gradient Boosting model, we uncover the exact non-linear behavioral and physical mechanisms driving customer attrition."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-s7-shap-beeswarm",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── SHAP Global Importance: Summary Bee-swarm Plot ─────────────────────────────\n",
            "print('Displaying Global SHAP Feature Importance Bee-swarm (Top 20 Drivers):')\n",
            "display(Image(filename=str(PATHS.figures / '06_shap_summary_beeswarm.png')))\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-s7-shap-tipping",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── SHAP Dependence Plot: Non-Linear Tipping Points ───────────────────────────\n",
            "print('Displaying Empirical Tipping Points (Device Age Obsolescence & Unit Cost Friction):')\n",
            "display(Image(filename=str(PATHS.figures / '07_shap_dependence_tipping_points.png')))\n"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "cell-s8-header",
        "metadata": {},
        "source": [
            "---\n",
            "## 8. Financial Decision Engine: 2D Risk-Value Matrix & Expected Profit Maximization\n",
            "\n",
            "### Economic Decision Theory in Banking & Telecommunications\n",
            "Standard machine learning models pick an arbitrary decision cutoff $p=0.50$, destroying shareholder value. The **TelecomFinancialDecisionEngine** optimizes the decision threshold $p^*$ by maximizing:\n",
            "\n",
            "$$\\text{Expected Net Profit}(p^*) = \\sum_{i \\in \\text{Targeted}} \\left[ R_k \\times \\text{LTV}_i - C_k \\right]$$\n",
            "\n",
            "Where customers are dynamically triaged into a **2D Risk-Value Strategic Matrix** based on their dual risk signals ($P(\\text{churn})$ and $P(\\text{compression})$)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-s8-decision-engine-run",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Initialize Financial Decision Engine & Optimize Threshold* ────────────────\n",
            "calibrated_model = benchmark_suite.calibrated_ensemble\n",
            "p_churn_test = calibrated_model.predict_proba(X_test_final)[:, 1]\n",
            "p_comp_test  = stage1_model.predict_proba(X_test_final)\n",
            "\n",
            "rev_mean_test = X_test['rev_Mean'].fillna(50.0).values if 'rev_Mean' in X_test else np.full(len(X_test), 50.0)\n",
            "eqpdays_test  = X_test['eqpdays'].fillna(340.0).values if 'eqpdays' in X_test else np.full(len(X_test), 340.0)\n",
            "\n",
            "decision_engine = TelecomFinancialDecisionEngine(\n",
            "    p_churn=p_churn_test,\n",
            "    p_comp=p_comp_test,\n",
            "    rev_mean=rev_mean_test,\n",
            "    eqpdays=eqpdays_test,\n",
            "    scale_factor=5.0\n",
            ")\n",
            "\n",
            "optimal_p, curve_df = decision_engine.optimize_threshold(p_comp_threshold=0.35)\n",
            "print(f'★ Optimal Economic Decision Threshold: p* = {optimal_p:.2f}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-s8-strategy-benchmark",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Executive Strategy Comparison (100,000 Customers Annual Scale) ─────────────\n",
            "strat_df = decision_engine.benchmark_strategies(optimal_p=optimal_p, p_comp_threshold=0.35)\n",
            "print('=' * 85)\n",
            "print('  STRATEGY FINANCIAL BENCHMARK (EXPECTED VALUE MAXIMIZATION)')\n",
            "print('=' * 85)\n",
            "display(strat_df)\n",
            "\n",
            "# Display Expected Profit Curve & 2D Risk-Value Matrix\n",
            "display(Image(filename=str(PATHS.figures / '08_profit_curve_and_2d_matrix.png')))\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-s8-final-takeaways",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Final Executive Takeaways & Recommendations ───────────────────────────────\n",
            "print('=' * 75)\n",
            "print('  FINAL QUANTITATIVE EXECUTIVE RECOMMENDATIONS')\n",
            "print('=' * 75)\n",
            "opt_net = strat_df.loc[3, 'Net Profit Lift ($M)']\n",
            "opt_cost = strat_df.loc[3, 'Campaign Cost ($M)']\n",
            "opt_roi = strat_df.loc[3, 'ROI (%)']\n",
            "opt_cust = strat_df.loc[3, 'Targeted Customers']\n",
            "print(f'1. Optimal Strategy Cutoff        : Threshold p* = {optimal_p:.2f}')\n",
            "print(f'2. Targeted High-Impact Customers : {opt_cust:,} customers (scaled to 100k)')\n",
            "print(f'3. Targeted Retention Budget      : ${opt_cost:.2f}M USD')\n",
            "print(f'4. Net Economic Profit Generated  : ${opt_net:.2f}M USD (Incremental gain)')\n",
            "print(f'5. Return on Retention Investment : {opt_roi:.1f}% ROI')\n",
            "print('=' * 75)\n"
        ]
    }
]

# Find index of cell-s6-header
idx_s6 = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("id") == "cell-s6-header":
        idx_s6 = i
        break

assert idx_s6 is not None, "Could not find cell-s6-header!"

# Replace all cells from idx_s6 onwards with new_phase3_4_cells
nb["cells"] = nb["cells"][:idx_s6] + new_phase3_4_cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully updated notebook with Phase 3 & 4! Total cells: {len(nb['cells'])}")
