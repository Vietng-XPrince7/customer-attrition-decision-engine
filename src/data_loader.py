"""
data_loader.py — Data Ingestion, Schema Validation & Merge Pipeline
=====================================================================
Handles reading Client.csv and Record.csv, validates schema integrity,
performs the join on Customer_ID, and returns a clean merged DataFrame
ready for feature engineering.

Usage
-----
    from src.data_loader import load_and_merge

    df = load_and_merge()
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from src.config import PATHS, COLUMNS, RANDOM_STATE

logger = logging.getLogger(__name__)


# =============================================================================
# Expected Schema
# =============================================================================
CLIENT_SCHEMA: dict[str, type] = {
    "Customer_ID":      (np.integer, object),
    "totcalls":         (np.integer, np.floating),
    "totrev":           np.floating,
    "avgrev":           np.floating,
    "income":           np.floating,
    "eqpdays":          np.floating,
}

RECORD_SCHEMA: dict[str, type] = {
    "Customer_ID":      (np.integer, object),
    "churn":            np.integer,
    "months":           np.integer,
    "rev_Mean":         np.floating,
    "mou_Mean":         np.floating,
}


# =============================================================================
# Public API
# =============================================================================
def load_and_merge(
    raw_dir:       Optional[Path] = None,
    client_file:   str = "Client.csv",
    record_file:   str = "Record.csv",
    validate:      bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Read, validate, and merge Client.csv and Record.csv.

    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing the raw CSV files. Defaults to ``PATHS.data_raw``.
    client_file : str
        Filename for the client demographic table.
    record_file : str
        Filename for the customer behaviour/record table.
    validate : bool
        If True, run schema integrity checks and log warnings on anomalies.
    drop_duplicates : bool
        If True, remove duplicate Customer_ID rows before merging.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns from both tables, indexed by
        Customer_ID (reset to RangeIndex).

    Raises
    ------
    FileNotFoundError
        If either CSV file cannot be found in ``raw_dir``.
    ValueError
        If the merge produces zero rows (likely a key mismatch).
    """
    raw_dir = raw_dir or PATHS.data_raw

    client_path = raw_dir / client_file
    record_path = raw_dir / record_file

    client_df = _read_csv(client_path, "Client")
    record_df = _read_csv(record_path, "Record")

    if validate:
        _validate_schema(client_df, CLIENT_SCHEMA, label="Client")
        _validate_schema(record_df, RECORD_SCHEMA, label="Record")
        _check_missing(client_df, label="Client")
        _check_missing(record_df, label="Record")

    if drop_duplicates:
        client_df = _drop_duplicate_ids(client_df, COLUMNS.id, label="Client")
        record_df = _drop_duplicate_ids(record_df, COLUMNS.id, label="Record")

    merged = _merge(client_df, record_df)

    logger.info("Merge complete. Final shape: %s", merged.shape)
    return merged


def load_processed(filename: str = "merged_clean.parquet") -> pd.DataFrame:
    """
    Load a previously saved processed dataset from ``PATHS.data_processed``.

    Parameters
    ----------
    filename : str
        Name of the Parquet file to load.

    Returns
    -------
    pd.DataFrame
    """
    path = PATHS.data_processed / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. "
            "Run load_and_merge() and save_processed() first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded processed data from %s. Shape: %s", path, df.shape)
    return df


def save_processed(df: pd.DataFrame, filename: str = "merged_clean.parquet") -> Path:
    """
    Persist the processed DataFrame to ``PATHS.data_processed`` as Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        The processed DataFrame to save.
    filename : str
        Output filename (Parquet format).

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    out_path = PATHS.data_processed / filename
    df.to_parquet(out_path, index=False)
    logger.info("Saved processed data → %s", out_path)
    return out_path


# =============================================================================
# Internal Helpers
# =============================================================================
def _read_csv(path: Path, label: str) -> pd.DataFrame:
    """Read a CSV file with error handling."""
    if not path.exists():
        raise FileNotFoundError(
            f"[{label}] CSV not found: {path}\n"
            f"Place your raw files in: {PATHS.data_raw}"
        )
    df = pd.read_csv(path)
    logger.info("[%s] Loaded %d rows × %d cols from %s", label, *df.shape, path.name)
    return df


def _validate_schema(df: pd.DataFrame, schema: dict, label: str) -> None:
    """
    Warn if expected columns are missing or have unexpected dtypes.

    Parameters
    ----------
    df : pd.DataFrame
    schema : dict
        Mapping of column_name → expected numpy dtype.
    label : str
        Human-readable table label for log messages.
    """
    missing_cols = [c for c in schema if c not in df.columns]
    if missing_cols:
        logger.warning("[%s] Missing expected columns: %s", label, missing_cols)

    for col, expected_dtype in schema.items():
        if col in df.columns:
            actual = df[col].dtype
            if isinstance(expected_dtype, tuple):
                valid = any(np.issubdtype(actual, t) for t in expected_dtype)
            else:
                valid = np.issubdtype(actual, expected_dtype)
            if not valid:
                logger.warning(
                    "[%s] Column '%s': expected %s, got %s",
                    label, col, expected_dtype, actual,
                )


def _check_missing(df: pd.DataFrame, label: str) -> None:
    """Log the percentage of missing values per column."""
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 0]
    if not high_missing.empty:
        logger.warning("[%s] Missing value rates:\n%s", label, high_missing.to_string())
    else:
        logger.info("[%s] No missing values detected.", label)


def _drop_duplicate_ids(df: pd.DataFrame, id_col: str, label: str) -> pd.DataFrame:
    """Remove duplicate rows based on the join key, keeping the first occurrence."""
    n_before = len(df)
    df = df.drop_duplicates(subset=[id_col], keep="first")
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("[%s] Dropped %d duplicate Customer_IDs.", label, n_dropped)
    return df


def _merge(client_df: pd.DataFrame, record_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the two tables on Customer_ID and validate the result."""
    merged = client_df.merge(record_df, on=COLUMNS.id, how="inner", suffixes=("", "_rec"))

    if len(merged) == 0:
        raise ValueError(
            "Merge produced 0 rows. Verify that Customer_ID values match "
            "between Client.csv and Record.csv."
        )

    # Report join quality
    n_client = len(client_df)
    n_match  = len(merged)
    logger.info(
        "Join quality: %d / %d Client records matched (%.1f%%)",
        n_match, n_client, 100 * n_match / n_client,
    )
    return merged.reset_index(drop=True)


def get_summary_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return descriptive statistics and missing-value report for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (describe_df, missing_report_df)
    """
    describe_df = df.describe(include="all").T
    missing_report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_pct":   df.isnull().mean() * 100,
        "dtype":         df.dtypes,
        "n_unique":      df.nunique(),
    })
    return describe_df, missing_report

