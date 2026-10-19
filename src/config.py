"""
config.py — Centralised Configuration & Constants
===================================================
Single source of truth for all paths, random seeds, column registries,
and financial hyper-parameters used across the project.

Usage
-----
    from src.config import PATHS, COLUMNS, FINANCIAL, RANDOM_STATE
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_STATE: int = 42

# =============================================================================
# Directory Paths
# =============================================================================
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class _Paths:
    """Immutable path registry for the project."""
    root:           Path = ROOT_DIR
    data_raw:       Path = ROOT_DIR / "data" / "raw"
    data_processed: Path = ROOT_DIR / "data" / "processed"
    notebooks:      Path = ROOT_DIR / "notebooks"
    reports:        Path = ROOT_DIR / "reports"
    figures:        Path = ROOT_DIR / "reports" / "figures"
    src:            Path = ROOT_DIR / "src"

    def __post_init__(self) -> None:
        """Ensure all output directories exist at import time."""
        for f in [self.data_processed, self.figures]:
            f.mkdir(parents=True, exist_ok=True)

PATHS: _Paths = _Paths()

# =============================================================================
# Column Registry
# =============================================================================
@dataclass(frozen=True)
class _Columns:
    """Column names shared across data and feature modules."""
    # ── Join key ──────────────────────────────────────────────────────────────
    id:                  str = "Customer_ID"

    # ── Client.csv identifiers ────────────────────────────────────────────────
    client_features: List[str] = field(default_factory=lambda: [
        "uniqsubs", "actvsubs", "new_cell", "crclscod", "asl_flag",
        "totcalls", "totmou", "totrev", "adjrev", "adjmou", "adjqty",
        "avgrev", "avgmou", "avgqty", "avg3mou", "avg3qty", "avg3rev",
        "avg6mou", "avg6qty", "avg6rev", "prizm_social_one", "area",
        "dualband", "refurb_new", "hnd_price", "phones", "models",
        "hnd_webcap", "truck", "rv", "ownrent", "lor", "dwlltype",
        "marital", "adults", "infobase", "income", "numbcars", "HHstatin",
        "dwllsize", "forgntvl", "ethnic", "kid0_2", "kid3_5", "kid6_10",
        "kid11_15", "kid16_17", "creditcd", "eqpdays",
    ])

    # ── Record.csv identifiers ────────────────────────────────────────────────
    record_features: List[str] = field(default_factory=lambda: [
        "rev_Mean", "mou_Mean", "totmrc_Mean", "da_Mean", "ovrmou_Mean",
        "ovrrev_Mean", "vceovr_Mean", "datovr_Mean", "roam_Mean",
        "change_mou", "change_rev", "drop_vce_Mean", "drop_dat_Mean",
        "blck_vce_Mean", "blck_dat_Mean", "unan_vce_Mean", "unan_dat_Mean",
        "plcd_vce_Mean", "plcd_dat_Mean", "recv_vce_Mean", "recv_sms_Mean",
        "comp_vce_Mean", "comp_dat_Mean", "custcare_Mean", "ccrndmou_Mean",
        "cc_mou_Mean", "inonemin_Mean", "threeway_Mean", "mou_cvce_Mean",
        "mou_cdat_Mean", "mou_rvce_Mean", "owylis_vce_Mean", "mouowylisv_Mean",
        "iwylis_vce_Mean", "mouiwylisv_Mean", "peak_vce_Mean", "peak_dat_Mean",
        "mou_peav_Mean", "mou_pead_Mean", "opk_vce_Mean", "opk_dat_Mean",
        "mou_opkv_Mean", "mou_opkd_Mean", "drop_blk_Mean", "attempt_Mean",
        "complete_Mean", "callfwdv_Mean", "callwait_Mean", "months",
    ])

    # ── Targets ───────────────────────────────────────────────────────────────
    target_churn:    str = "churn"           # Binary: 0 / 1
    target_revenue:  str = "RevenueLoss"     # Continuous (engineered)

    # ── Feature groups ────────────────────────────────────────────────────────
    categorical: List[str] = field(default_factory=lambda: [
        "new_cell", "crclscod", "asl_flag", "prizm_social_one", "area",
        "dualband", "refurb_new", "hnd_webcap", "ownrent", "dwlltype",
        "marital", "infobase", "HHstatin", "dwllsize", "ethnic", "creditcd",
    ])
    binary: List[str] = field(default_factory=lambda: [
        "truck", "rv", "forgntvl", "kid0_2", "kid3_5", "kid6_10", "kid11_15", "kid16_17",
    ])

COLUMNS: _Columns = _Columns()

# =============================================================================
# Financial Parameters  (basis for Expected Profit computation)
# =============================================================================
@dataclass(frozen=True)
class _Financial:
    """
    Business parameters used in the Expected Profit Framework.

    Notation
    --------
    CLV_mean            : Average Customer Lifetime Value (USD)
    retention_cost      : Cost of a targeted retention intervention (USD)
    false_alarm_cost    : Opportunity cost of treating a loyal customer (USD)
    revenue_compression : Fraction of CLV lost when a customer downgrades
                          without churning (Revenue Compression scenario)
    """
    CLV_mean:             float = 5_000.0   # USD — adjust to actual data
    retention_cost:       float =   250.0   # USD per outreach action
    false_alarm_cost:     float =    30.0   # USD (discount / voucher cost)
    revenue_compression:  float =     0.35  # 35 % revenue loss proxy

    # ── Profit matrix entries ─────────────────────────────────────────────────
    @property
    def TP_benefit(self) -> float:
        """True Positive: churn predicted, customer saved → net CLV retained."""
        return self.CLV_mean - self.retention_cost

    @property
    def FP_cost(self) -> float:
        """False Positive: loyal customer receives unnecessary retention offer."""
        return -self.false_alarm_cost

    @property
    def FN_cost(self) -> float:
        """False Negative: churner missed — full CLV lost."""
        return -self.CLV_mean

    @property
    def TN_benefit(self) -> float:
        """True Negative: correctly identified loyal customer, no action."""
        return 0.0

FINANCIAL: _Financial = _Financial()

# =============================================================================
# Model Training Configuration
# =============================================================================
CV_FOLDS:         int   = 5
TEST_SIZE:        float = 0.20
CALIBRATION_CV:   int   = 5
OPTUNA_TRIALS:    int   = 100
OPTUNA_TIMEOUT:   int   = 600   # seconds

# =============================================================================
# SHAP Configuration
# =============================================================================
SHAP_MAX_DISPLAY:   int = 20
SHAP_SAMPLE_SIZE:   int = 2_000   # rows used for approximate SHAP on large sets

