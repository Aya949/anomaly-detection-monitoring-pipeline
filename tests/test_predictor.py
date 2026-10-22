"""
Unit tests for the API predictor.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_records(n: int = 50) -> list[dict]:
    """Generate n valid records for predictor input."""
    timestamps = pd.date_range(
        start="2014-02-14 14:30:00",
        periods=n,
        freq="5min"
    )
    return [
        {
            "timestamp"  : str(ts),
            "cpu_util_1" : float(np.random.uniform(0.1, 0.5)),
            "cpu_util_2" : float(np.random.uniform(1.0, 10.0)),
            "rds_cpu"    : float(np.random.uniform(5.0, 8.0)),
        }
        for ts in timestamps
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestAnomalyPredictor:

    @pytest.fixture
    def predictor(self):
        """Load the real predictor — requires models/ to exist."""
        from api.predictor import AnomalyPredictor
        return AnomalyPredictor()

    def test_model_loads_successfully(self, predictor):
        assert predictor.model  is not None
        assert predictor.scaler is not None
        assert predictor.feature_cols is not None
        assert predictor.config is not None

    def test_feature_cols_count(self, predictor):
        assert len(predictor.feature_cols) == 30

    def test_predict_returns_correct_keys(self, predictor):
        records = make_records(50)
        result  = predictor.predict(records)

        assert "total_records"     in result
        assert "anomaly_count"     in result
        assert "anomaly_rate_pct"  in result
        assert "threshold_used"    in result
        assert "predictions"       in result

    def test_predict_returns_correct_types(self, predictor):
        records = make_records(50)
        result  = predictor.predict(records)

        assert isinstance(result["total_records"],    int)
        assert isinstance(result["anomaly_count"],    int)
        assert isinstance(result["anomaly_rate_pct"], float)
        assert isinstance(result["predictions"],      list)

    def test_predict_anomaly_count_consistent(self, predictor):
        records = make_records(50)
        result  = predictor.predict(records)

        flagged = sum(
            1 for p in result["predictions"]
            if p["is_anomaly"] == 1
        )
        assert flagged == result["anomaly_count"]

    def test_predict_scores_are_positive(self, predictor):
        records = make_records(50)
        result  = predictor.predict(records)

        for p in result["predictions"]:
            assert p["anomaly_score"] > 0

    def test_predict_confidence_valid_values(self, predictor):
        records = make_records(50)
        result  = predictor.predict(records)

        valid = {"low", "medium", "high"}
        for p in result["predictions"]:
            assert p["confidence"] in valid

    def test_too_few_rows_raises_error(self, predictor):
        records = make_records(10)  # less than WINDOW_LONG
        with pytest.raises(ValueError):
            predictor.predict(records)

    def test_anomaly_rate_matches_count(self, predictor):
        records = make_records(50)
        result  = predictor.predict(records)

        expected_rate = round(
            result["anomaly_count"] / result["total_records"] * 100, 2
        )
        assert result["anomaly_rate_pct"] == expected_rate