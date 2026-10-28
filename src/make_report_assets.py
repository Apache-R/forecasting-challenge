"""Generate PNG figures and public-holdout metrics for the final report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_and_aggregate
from evaluate import evaluate_public_holdout

def generate_assets(
    data_dir: str | Path = "challenge_data",
    predictions_file: str | Path = "predictions.csv",
    output_dir: str | Path = "reports/figures",
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_and_aggregate(data_dir)
    predictions = pd.read_csv(predictions_file)
    train = frame.iloc[:110]
    holdout = frame.iloc[110:]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(frame["timestamp"] / 60, frame["active_vm_count"], color="#2563eb")
    axes[0].axvline(train.iloc[-1]["timestamp"] / 60, color="#dc2626", linestyle="--")
    axes[0].set_ylabel("Active VMs")
    axes[0].set_title("Aggregate target series and 110/24 split")
    axes[1].plot(
        frame["timestamp"] / 60,
        frame["aggregate_cpu_utilization"],
        color="#059669",
    )
    axes[1].axvline(train.iloc[-1]["timestamp"] / 60, color="#dc2626", linestyle="--")
    axes[1].set_xlabel("Minutes from trace start")
    axes[1].set_ylabel("Mean CPU utilization (%)")
    fig.tight_layout()
    fig.savefig(output / "target_series.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        holdout["timestamp"] / 60,
        holdout["aggregate_cpu_utilization"],
        label="Observed holdout",
        color="#111827",
        marker="o",
    )
    ax.plot(
        predictions["timestamp"] / 60,
        predictions["aggregate_cpu_utilization"],
        label="Ridge forecast",
        color="#2563eb",
        marker="o",
    )
    ax.plot(
        holdout["timestamp"] / 60,
        [train.iloc[-1]["aggregate_cpu_utilization"]] * len(holdout),
        label="Persistence baseline",
        color="#dc2626",
        linestyle="--",
    )
    ax.set_title("Final 2-hour public holdout: CPU forecast versus actual")
    ax.set_xlabel("Minutes from trace start")
    ax.set_ylabel("Aggregate CPU utilization (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "holdout_forecast.png", dpi=160)
    plt.close(fig)

    scores = evaluate_public_holdout(data_dir, predictions_file)
    scores.to_csv(output.parent / "public_holdout_scores.csv", index=False)
    cpu = scores[scores["target"] == "aggregate_cpu_utilization"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        ["Persistence", "Ridge model"],
        [cpu["baseline_mae"], cpu["mae"]],
        color=["#dc2626", "#2563eb"],
    )
    ax.set_title("Public holdout CPU MAE")
    ax.set_ylabel("MAE (percentage points)")
    for i, value in enumerate([cpu["baseline_mae"], cpu["mae"]]):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output / "holdout_mae.png", dpi=160)
    plt.close(fig)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--predictions", default="predictions.csv")
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()
    scores = generate_assets(args.data, args.predictions, args.output_dir)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
