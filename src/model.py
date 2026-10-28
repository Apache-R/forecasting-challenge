"""Forecasting model definitions and multi-step inference."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Ensures 'src' is in sys.path so 'features' and 'cwfc' can be imported
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import create_cpu_features


class WorkloadForecaster:
    """Hybrid forecaster:

    - Active VM Count: Persistence baseline (optimal for zero-churn cohorts).
    - Aggregate CPU: Autoregressive Ridge Regression with recursive rollout.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.cpu_model = Ridge(alpha=self.alpha)
        self.last_known_vm_count: float = 0.0
        self.train_cpu_history: list[float] = []

    def fit(self, train_df: pd.DataFrame) -> WorkloadForecaster:
        """Fits the CPU model and captures fleet parameters."""
        self.last_known_vm_count = float(train_df["active_vm_count"].iloc[-1])

        train_series = train_df["aggregate_cpu_utilization"].copy()
        self.train_cpu_history = list(train_series.values)

        features_df = create_cpu_features(train_series)
        valid_idx = features_df.dropna().index

        X_train = features_df.loc[valid_idx].drop(columns=["cpu"])
        y_train = features_df.loc[valid_idx, "cpu"]

        self.cpu_model.fit(X_train, y_train)
        return self

    def predict(self, horizon: int = 24) -> tuple[list[float], list[float]]:
        """Produces multi-step recursive forecasts bounded to valid physical ranges."""
        history = list(self.train_cpu_history)
        cpu_forecasts: list[float] = []

        for _ in range(horizon):
            curr_series = pd.Series(history)
            curr_feats = (
                create_cpu_features(curr_series).iloc[[-1]].drop(columns=["cpu"])
            )

            pred_next = self.cpu_model.predict(curr_feats)[0]
            pred_next = float(np.clip(pred_next, 0.0, 100.0))

            cpu_forecasts.append(pred_next)
            history.append(pred_next)

        vm_forecasts = [max(0.0, self.last_known_vm_count)] * horizon
        
        return vm_forecasts, cpu_forecasts