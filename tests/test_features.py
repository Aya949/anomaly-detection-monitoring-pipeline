"""
Unit tests for feature engineering.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features import (
    engineer_features,
    get_feature_cols,
    VALUE_COLS,
    WINDOW_SHORT,
    WINDOW_LONG,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_df(n_rows: int = 100) -> pd.DataFrame:
    """Create a minimal valid DataFrame for testing."""
    timestamps = pd.date_range(
        start="2014-02-14 14:30:00",
        periods=n_rows,
        freq="5min"
    )
    return pd.DataFrame({
        "cpu_util_1" : np.random.uniform(0.1, 0.5, n_rows),
        "cpu_util_2" : np.random.uniform(1.0, 10.0, n_rows),
        "rds_cpu"    : np.random.uniform(5.0, 8.0, n_rows),
        "is_anomaly" : np.zeros(n_rows, dtype=int),
    }, index=timestamps)


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestEngineerFeatures:

    def test_output_has_expected_columns(self):
        df   = make_df(100)
        feat = engineer_features(df)

        for col in VALUE_COLS:
            assert f"{col}_mean_s" in feat.columns
            assert f"{col}_std_s"  in feat.columns
            assert f"{col}_min_s"  in feat.columns
            assert f"{col}_max_s"  in feat.columns
            assert f"{col}_mean_l" in feat.columns
            assert f"{col}_std_l"  in feat.columns
            assert f"{col}_zscore" in feat.columns
            assert f"{col}_diff"   in feat.columns
            assert f"{col}_diff2"  in feat.columns

    def test_output_has_27_feature_columns(self):
        df       = make_df(100)
        feat     = engineer_features(df)
        feat_cols= get_feature_cols(feat)
        # 9 engineered features × 3 metrics + 3 raw cols = 30
        assert len(feat_cols) == 30

    def test_nan_rows_are_dropped(self):
        df   = make_df(100)
        feat = engineer_features(df)
        assert feat.isnull().sum().sum() == 0

    def test_rows_lost_equals_window_long(self):
        n    = 100
        df   = make_df(n)
        feat = engineer_features(df)
        # Loses WINDOW_LONG + 2 rows for rolling windows and diff2
        # 100 - 36 + 1 (rolling min) - 2 (diff2) = 65
        assert len(feat) == 65

    def test_missing_column_raises_error(self):
        df = make_df(100).drop(columns=["cpu_util_1"])
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer_features(df)

    def test_zscore_is_zero_for_constant_series(self):
        df = make_df(100)
        df["cpu_util_1"] = 5.0  # constant
        feat = engineer_features(df)
        # Z-score of constant series should be ~0
        assert feat["cpu_util_1_zscore"].abs().max() < 1e-5

    def test_feature_values_are_finite(self):
        df   = make_df(100)
        feat = engineer_features(df)
        feat_cols = get_feature_cols(feat)
        assert np.isfinite(feat[feat_cols].values).all()


class TestGetFeatureCols:

    def test_excludes_is_anomaly(self):
        df       = make_df(100)
        feat     = engineer_features(df)
        feat_cols= get_feature_cols(feat)
        assert "is_anomaly" not in feat_cols

    def test_excludes_is_anomaly_per_feature(self):
        df = make_df(100)
        df["is_anomaly_cpu_util_1"] = 0
        feat      = engineer_features(df)
        feat_cols = get_feature_cols(feat)
        assert "is_anomaly_cpu_util_1" not in feat_cols

    def test_includes_all_value_cols(self):
        df       = make_df(100)
        feat     = engineer_features(df)
        feat_cols= get_feature_cols(feat)
        for col in VALUE_COLS:
            assert col in feat_cols