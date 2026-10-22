import joblib
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Works in both local and Docker environments
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))


# ── constants (must match notebook) ───────────────────────────────────────────
WINDOW_SHORT = 6
WINDOW_LONG  = 36
VALUE_COLS   = ["cpu_util_1", "cpu_util_2", "rds_cpu"]


class AnomalyPredictor:

    def __init__(self):
        self.model        = None
        self.scaler       = None
        self.feature_cols = None
        self.config       = None
        self._load()

    def _load(self):
        self.model  = joblib.load(MODELS_DIR / "isolation_forest.pkl")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")

        with open(MODELS_DIR / "feature_cols.json") as f:
            self.feature_cols = json.load(f)

        with open(MODELS_DIR / "model_config.json") as f:
            self.config = json.load(f)

        print(f"[predictor] model loaded — "
              f"{len(self.feature_cols)} features | "
              f"threshold: {self.config['best_threshold']:.4f}")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = df[VALUE_COLS].copy()

        for col in VALUE_COLS:
            feat[f"{col}_mean_s"] = df[col].rolling(WINDOW_SHORT).mean()
            feat[f"{col}_std_s"]  = df[col].rolling(WINDOW_SHORT).std()
            feat[f"{col}_min_s"]  = df[col].rolling(WINDOW_SHORT).min()
            feat[f"{col}_max_s"]  = df[col].rolling(WINDOW_SHORT).max()
            feat[f"{col}_mean_l"] = df[col].rolling(WINDOW_LONG).mean()
            feat[f"{col}_std_l"]  = df[col].rolling(WINDOW_LONG).std()
            feat[f"{col}_zscore"] = (
                (df[col] - feat[f"{col}_mean_l"]) /
                (feat[f"{col}_std_l"] + 1e-8)
            )
            feat[f"{col}_diff"]  = df[col].diff()
            feat[f"{col}_diff2"] = df[col].diff().diff()

        return feat.dropna()

    def _confidence(self, score: float, threshold: float) -> str:
        if score < threshold * 0.9:
            return "low"
        elif score < threshold * 1.1:
            return "medium"
        else:
            return "high"

    def predict(self, records: list) -> dict:
        # Build dataframe
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Engineer features
        df_feat = self._engineer_features(df)

        if df_feat.empty:
            raise ValueError(
                f"Not enough data after feature engineering. "
                f"Send at least {WINDOW_LONG + 5} rows."
            )

        # Align to training feature order
        X        = df_feat[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        # Score
        anomaly_scores = -self.model.score_samples(X_scaled)
        threshold      = self.config["best_threshold"]
        predictions    = (anomaly_scores >= threshold).astype(int)

        # Build results
        results = []
        for i, (ts, row) in enumerate(df_feat.iterrows()):
            results.append({
                "timestamp"     : str(ts),
                "cpu_util_1"    : round(float(row["cpu_util_1"]), 4),
                "cpu_util_2"    : round(float(row["cpu_util_2"]), 4),
                "rds_cpu"       : round(float(row["rds_cpu"]), 4),
                "anomaly_score" : round(float(anomaly_scores[i]), 4),
                "is_anomaly"    : int(predictions[i]),
                "confidence"    : self._confidence(
                                    anomaly_scores[i], threshold),
            })

        anomaly_count = int(predictions.sum())

        return {
            "total_records"    : len(results),
            "anomaly_count"    : anomaly_count,
            "anomaly_rate_pct" : round(anomaly_count / len(results) * 100, 2),
            "threshold_used"   : round(threshold, 4),
            "predictions"      : results,
        }

    def save_predictions_to_db(self, result: dict) -> None:
        """Save prediction results to PostgreSQL."""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host    ="postgres",
                port    = 5432,
                dbname  = "anomaly_db",
                user    = "airflow",
                password= os.getenv("POSTGRES_PASSWORD"),
            )
            cursor = conn.cursor()

            for p in result["predictions"]:
                cursor.execute("""
                    INSERT INTO marts.anomaly_predictions
                        (timestamp, cpu_util_1, cpu_util_2, rds_cpu,
                        anomaly_score, is_anomaly, confidence, threshold_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    p["timestamp"],
                    p["cpu_util_1"],
                    p["cpu_util_2"],
                    p["rds_cpu"],
                    p["anomaly_score"],
                    p["is_anomaly"],
                    p["confidence"],
                    result["threshold_used"],
                ))

            conn.commit()
            cursor.close()
            conn.close()
            print(f"[predictor] Saved {len(result['predictions'])} predictions to DB")

        except Exception as e:
            print(f"[predictor] DB save failed (non-critical): {e}")
# Singleton — loaded once at startup
predictor = AnomalyPredictor()