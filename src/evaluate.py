"""Leakage-safe rolling-origin evaluation for the forecasting pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_and_aggregate
from model import WorkloadForecaster
from cwfc.baselines import persistence_forecast
from cwfc.metrics import mae, normalized_mae, rmse, smape
from cwfc.validation import rolling_origin_splits

def evaluate(data_dir: str | Path, validation_horizon: int = 24) -> pd.DataFrame:
    """Evaluate both targets using expanding temporal validation.

    Only timestamps before each validation window are used for fitting.
    The final 24 observations remain untouched by this evaluation.
    """
    frame = load_and_aggregate(data_dir)
    frame_for_validation = frame.copy()
    frame_for_validation["timestamp_dt"] = pd.to_datetime(frame_for_validation["timestamp"], unit="s", utc=True)

    model_rows: list[dict[str, object]] = []
    final_holdout_start = len(frame) - 24
    min_train = max(62, validation_horizon * 2)

    validation_splits = rolling_origin_splits(
        frame_for_validation.iloc[:final_holdout_start],
        timestamp_col="timestamp_dt",
        horizon=validation_horizon,
        min_train_size=min_train,
        step=validation_horizon // 2,
    )

    for split in validation_splits:
        train_mask = frame_for_validation["timestamp_dt"] <= split.train_end

        validation_mask = (
            (frame_for_validation["timestamp_dt"] >= split.validation_start)
            & (frame_for_validation["timestamp_dt"] <= split.validation_end)
        )

        train = frame_for_validation.loc[train_mask].drop(columns=["timestamp_dt"]).reset_index(drop=True)
        validation = frame_for_validation.loc[validation_mask].drop(columns=["timestamp_dt"]).reset_index(drop=True)
        forecaster = WorkloadForecaster(alpha=1.0).fit(train)
        model_vm, model_cpu = forecaster.predict(validation_horizon)

        for target, predictions in (
            ("active_vm_count", model_vm),
            ("aggregate_cpu_utilization", model_cpu),
        ):
            actual = validation[target].to_numpy()
            predictions_array = np.asarray(predictions, dtype=float)
            baseline = persistence_forecast(train, validation, target).to_numpy()
            peak_threshold = train[target].quantile(0.90)
            peak_mask = actual >= peak_threshold

            model_rows.append(
                {
                    "origin": len(train),
                    "target": target,
                    "mae": mae(actual, predictions_array),
                    "rmse": rmse(actual, predictions_array),
                    "smape": smape(actual, predictions_array),
                    "baseline_mae": mae(actual, baseline),
                    "normalized_mae": normalized_mae(
                        actual, predictions_array, baseline
                    ),
                    "peak_mae": (
                        mae(actual[peak_mask], predictions_array[peak_mask])
                        if peak_mask.any()
                        else float("nan")
                    ),
                }
            )

    if not model_rows:
        raise ValueError("Not enough pre-holdout history for temporal validation")
    
    return pd.DataFrame(model_rows)

def evaluate_public_holdout(
    data_dir: str | Path, prediction_file: str | Path
) -> pd.DataFrame:
    """Score an already-generated prediction file against the public holdout.

    This is an analysis-only score because the public holdout labels are
    available in this challenge release. It must not be used for model
    selection or tuning.
    """

    frame = load_and_aggregate(data_dir)
    predictions = pd.read_csv(prediction_file)
    expected = frame.iloc[-24:].reset_index(drop=True)

    if list(predictions.columns) != ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]:
        raise ValueError("Prediction file has the wrong columns")
    
    if len(predictions) != 24 or predictions["timestamp"].tolist() != expected["timestamp"].tolist():
        raise ValueError("Prediction file does not match the final 24 timestamps")

    rows = []
    for target in ("active_vm_count", "aggregate_cpu_utilization"):
        actual = expected[target].to_numpy(dtype=float)
        predicted = predictions[target].to_numpy(dtype=float)
        baseline = np.repeat(frame.iloc[109][target], 24).astype(float)
        rows.append(
            {
                "target": target,
                "mae": mae(actual, predicted),
                "rmse": rmse(actual, predicted),
                "smape": smape(actual, predicted),
                "baseline_mae": mae(actual, baseline),
                "normalized_mae": normalized_mae(actual, predicted, baseline),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--public-holdout",
        action="store_true",
        help="Score an existing prediction file against the public labels.",
    )
    parser.add_argument("--predictions", default="predictions.csv")
    args = parser.parse_args()

    if args.public_holdout:
        results = evaluate_public_holdout(args.data, args.predictions)
    else:
        results = evaluate(args.data)

    print(results.to_string(index=False))
    
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)


if __name__ == "__main__":
    main()
