"""Feature engineering for time-series forecasting."""

from __future__ import annotations

import pandas as pd


def create_cpu_features(series: pd.Series) -> pd.DataFrame:
    """Creates autoregressive lag and rolling window features from a univariate CPU series."""
    df_feat = pd.DataFrame({"cpu": series})

    # Historical Lags (5, 10, 15, 30 minutes)
    df_feat["lag_1"] = df_feat["cpu"].shift(1)
    df_feat["lag_2"] = df_feat["cpu"].shift(2)
    df_feat["lag_3"] = df_feat["cpu"].shift(3)
    df_feat["lag_6"] = df_feat["cpu"].shift(6)

    # Rolling statistics over past 30 minutes (6 steps)
    df_feat["rolling_mean_6"] = (
        df_feat["cpu"].shift(1).rolling(window=6).mean()
    )
    df_feat["rolling_std_6"] = df_feat["cpu"].shift(1).rolling(window=6).std()

    # Momentum (recent rate of change)
    df_feat["diff_1"] = df_feat["lag_1"] - df_feat["lag_2"]
    df_feat["diff_3"] = df_feat["lag_1"] - df_feat["lag_3"]

    return df_feat