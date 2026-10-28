"""CLI entrypoint to generate submission predictions."""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Ensures 'src' is in sys.path so 'data', 'model', and 'cwfc' can be imported
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_and_aggregate, split_train_holdout
from model import WorkloadForecaster

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cloud workload predictions."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="challenge_data",
        help="Path to folder containing raw challenge data",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to output predictions CSV",
    )

    return parser.parse_args()

def validate_submission(df: pd.DataFrame, expected_timestamps=None):
    """Enforces all criteria in the Submission."""

    assert len(df) == 24, f"Expected 24 rows, got {len(df)}"
    expected_cols = [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]

    assert (
        list(df.columns) == expected_cols
    ), f"Columns must match {expected_cols}"

    assert df.isna().sum().sum() == 0, "Null or NaN values detected"

    assert np.isfinite(df[["active_vm_count", "aggregate_cpu_utilization"]]).all().all(), (
        "Predictions must be finite numeric values"
    )

    assert df["timestamp"].is_unique, "Timestamps must be unique"

    assert (df["active_vm_count"] >= 0).all(), "active_vm_count must be >= 0"

    assert (
        (df["aggregate_cpu_utilization"] >= 0)
        & (df["aggregate_cpu_utilization"] <= 100)
    ).all(), "aggregate_cpu_utilization must be in [0, 100]"

    assert df["timestamp"].is_monotonic_increasing, (
        "Timestamps must be sorted chronologically"
    )

    if expected_timestamps is not None:
        expected = pd.Series(expected_timestamps).reset_index(drop=True)
        actual = df["timestamp"].reset_index(drop=True)

        assert actual.equals(expected), "Timestamps must match the holdout exactly"

def main():
    args = parse_args()

    print(f"Loading data from: {args.data}")
    df_ts = load_and_aggregate(args.data)

    train_df, holdout_df = split_train_holdout(df_ts, holdout_size=24)

    print(f"Fitting model on {len(train_df)} training intervals...")
    forecaster = WorkloadForecaster(alpha=1.0)
    forecaster.fit(train_df)

    print("Generating 24-step forecast for holdout horizon...")
    vm_preds, cpu_preds = forecaster.predict(horizon=24)

    submission = pd.DataFrame(
        {
            "timestamp": holdout_df["timestamp"].values,
            "active_vm_count": vm_preds,
            "aggregate_cpu_utilization": cpu_preds,
        }
    )

    validate_submission(submission, expected_timestamps=holdout_df["timestamp"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved valid predictions to {output_path.resolve()}")

if __name__ == "__main__":
    main()