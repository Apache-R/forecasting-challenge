"""Data ingestion, validation, and target aggregation."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


CPU_COLUMNS = {"timestamp", "vm_id", "avg_cpu"}


def load_and_aggregate(data_dir: str | Path) -> pd.DataFrame:
    """Loads raw VM CPU readings, validates against gaps/duplicates,

    and produces the aggregate 5-minute target time series.
    """
    data_path = Path(data_dir)

    cpu_gz = data_path / "vm_cpu_readings.csv.gz"
    cpu_csv = data_path / "vm_cpu_readings.csv"

    if cpu_gz.exists():
        raw_file = cpu_gz
    elif cpu_csv.exists():
        raw_file = cpu_csv
    else:
        raise FileNotFoundError(
            f"Could not find vm_cpu_readings in {data_path.resolve()}"
        )

    df_raw = pd.read_csv(raw_file)
    missing_columns = CPU_COLUMNS.difference(df_raw.columns)
    if missing_columns:
        raise ValueError(
            f"CPU readings are missing required columns: {sorted(missing_columns)}"
        )

    # Invalid readings cannot contribute to either required target.
    df_valid = df_raw.dropna(subset=["timestamp", "vm_id", "avg_cpu"]).copy()
    numeric_cpu = pd.to_numeric(df_valid["avg_cpu"], errors="coerce")

    if numeric_cpu.isna().any():
        raise ValueError("avg_cpu contains non-numeric values")
    
    df_valid["avg_cpu"] = numeric_cpu

    # Deduplicate (timestamp, vm_id) pairs if present.
    df_valid = df_valid.drop_duplicates(subset=["timestamp", "vm_id"])

    # Aggregate to cloud-wide targets.
    df_ts = (
        df_valid.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Verify the source is on the required five-minute grid.
    time_diffs = df_ts["timestamp"].diff().dropna().unique()
    if len(time_diffs) and (
        len(time_diffs) != 1 or time_diffs[0] != 300
    ):
        raise ValueError(
            f"Expected a regular 300-second grid, found intervals: {time_diffs}"
        )

    return df_ts


def split_train_holdout( df: pd.DataFrame, holdout_size: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits target series chronologically without data leakage."""
    
    if len(df) <= holdout_size:
        raise ValueError(
            f"Series length ({len(df)}) must exceed holdout ({holdout_size})"
        )
    
    train_df = df.iloc[:-holdout_size].copy().reset_index(drop=True)
    holdout_df = df.iloc[-holdout_size:].copy().reset_index(drop=True)

    return train_df, holdout_df