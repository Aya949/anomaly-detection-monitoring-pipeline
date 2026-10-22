"""
Feature engineering for anomaly detection.
All transformations must match exactly what was used during training.
"""
import pandas as pd
import numpy as np
from typing import List

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SHORT : int       = 6
WINDOW_LONG  : int       = 36
VALUE_COLS   : List[str] = ["cpu_util_1", "cpu_util_2", "rds_cpu"]


def resample_to_5min(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Resample each raw series to 5-minute intervals.
    Required because NAB series have misaligned timestamps.

    Args:
        raw: dict of {name: DataFrame} with DatetimeIndex

    Returns:
        dict of {name: DataFrame} resampled to 5min
    """
    resampled = {}
    for name, df in raw.items():
        df_r = df[[name]].resample("5min").mean()
        if "is_anomaly" in df.columns:
            df_r[f"is_anomaly_{name}"] = df["is_anomaly"].resample("5min").max()
        resampled[name] = df_r
    return resampled


def merge_series(resampled: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple resampled series into one aligned DataFrame.

    Args:
        resampled: dict of {name: DataFrame} with aligned 5min timestamps

    Returns:
        Merged DataFrame with global is_anomaly column
    """
    names = list(resampled.keys())

    df_values = pd.concat(
        [resampled[n][n] for n in names],
        axis=1, join="inner"
    )

    flag_cols = [f"is_anomaly_{n}" for n in names
                 if f"is_anomaly_{n}" in resampled[n].columns]

    if flag_cols:
        df_flags = pd.concat(
            [resampled[n][f"is_anomaly_{n}"] for n in names
             if f"is_anomaly_{n}" in resampled[n].columns],
            axis=1, join="inner"
        )
        df = pd.concat([df_values, df_flags], axis=1)
        df["is_anomaly"] = df[flag_cols].max(axis=1)
    else:
        df = df_values

    return df.dropna()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal features from raw server metrics.

    Features per metric (9 total × 3 metrics = 27 features):
        - Rolling mean  (short + long window)
        - Rolling std   (short + long window)
        - Rolling min   (short window)
        - Rolling max   (short window)
        - Z-score       (deviation from long rolling mean)
        - Diff          (rate of change)
        - Diff²         (acceleration)

    Args:
        df: DataFrame with columns cpu_util_1, cpu_util_2, rds_cpu
            and DatetimeIndex sorted ascending

    Returns:
        DataFrame with all engineered features, NaN rows dropped
    """
    if not all(col in df.columns for col in VALUE_COLS):
        missing = [c for c in VALUE_COLS if c not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    feat = df.copy()

    for col in VALUE_COLS:
        # Short window features (30 min)
        feat[f"{col}_mean_s"] = df[col].rolling(WINDOW_SHORT).mean()
        feat[f"{col}_std_s"]  = df[col].rolling(WINDOW_SHORT).std()
        feat[f"{col}_min_s"]  = df[col].rolling(WINDOW_SHORT).min()
        feat[f"{col}_max_s"]  = df[col].rolling(WINDOW_SHORT).max()

        # Long window features (3 hours)
        feat[f"{col}_mean_l"] = df[col].rolling(WINDOW_LONG).mean()
        feat[f"{col}_std_l"]  = df[col].rolling(WINDOW_LONG).std()

        # Z-score: how many stds from local mean
        feat[f"{col}_zscore"] = (
            (df[col] - feat[f"{col}_mean_l"]) /
            (feat[f"{col}_std_l"] + 1e-8)
        )

        # Rate of change
        feat[f"{col}_diff"]  = df[col].diff()
        feat[f"{col}_diff2"] = df[col].diff().diff()

    return feat.dropna()


def get_feature_cols(df_feat: pd.DataFrame) -> List[str]:
    """
    Return only the feature columns (excludes label columns).

    Args:
        df_feat: DataFrame output of engineer_features()

    Returns:
        List of feature column names
    """
    exclude = {"is_anomaly"} | {
        c for c in df_feat.columns
        if c.startswith("is_anomaly_")
    }
    return [c for c in df_feat.columns if c not in exclude]