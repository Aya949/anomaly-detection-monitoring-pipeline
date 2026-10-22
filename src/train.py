"""
Training script for the Isolation Forest anomaly detection model.
Run directly: python -m src.train
"""
import os
import json
import joblib
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, roc_auc_score
)
import numpy as np

from src.features import engineer_features, get_feature_cols

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "nab_aws_merged.csv"
MODELS_DIR = BASE_DIR / "models"


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and validate the merged dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    print(f"[train] Loaded {len(df)} rows from {data_path.name}")
    return df


def tune_threshold(
    anomaly_scores: np.ndarray,
    y_true: np.ndarray
) -> tuple[float, dict]:
    """
    Find the best classification threshold by maximizing F1.

    Args:
        anomaly_scores: continuous anomaly scores from model
        y_true: binary ground truth labels

    Returns:
        Tuple of (best_threshold, best_metrics_dict)
    """
    thresholds = np.percentile(anomaly_scores, np.arange(80, 99, 0.5))
    best_f1    = 0
    best_thresh= thresholds[0]
    best_metrics = {}

    for thresh in thresholds:
        y_pred = (anomaly_scores >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                "precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall"    : round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1"        : round(f1, 4),
            }

    return float(best_thresh), best_metrics


def train(
    n_estimators : int   = 200,
    contamination: float = None,
    run_name     : str   = "isolation_forest",
) -> dict:
    """
    Full training pipeline:
      1. Load data
      2. Engineer features
      3. Scale
      4. Train Isolation Forest
      5. Tune threshold
      6. Log to MLflow
      7. Save artifacts

    Returns:
        dict with model metrics and config
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    mlflow.set_experiment("nab-anomaly-detection")

    # ── Data ──────────────────────────────────────────────────────────────────
    df      = load_data(DATA_PATH)
    df_feat = engineer_features(df)

    feature_cols = get_feature_cols(df_feat)
    X = df_feat[feature_cols].values
    y = df_feat["is_anomaly"].values

    if contamination is None:
        contamination = round(float(y.mean()), 4)

    print(f"[train] {len(feature_cols)} features | "
          f"contamination: {contamination:.4f} | "
          f"anomaly rate: {y.mean()*100:.2f}%")

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train ─────────────────────────────────────────────────────────────────
    params = {
        "n_estimators"  : n_estimators,
        "contamination" : contamination,
        "max_samples"   : "auto",
        "random_state"  : 42,
    }

    with mlflow.start_run(run_name=run_name):
        model = IsolationForest(**params)
        model.fit(X_scaled)

        anomaly_scores = -model.score_samples(X_scaled)
        roc_auc        = roc_auc_score(y, anomaly_scores)

        best_threshold, best_metrics = tune_threshold(anomaly_scores, y)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc",   round(roc_auc, 4))
        mlflow.log_metric("precision", best_metrics["precision"])
        mlflow.log_metric("recall",    best_metrics["recall"])
        mlflow.log_metric("f1",        best_metrics["f1"])
        mlflow.log_metric("threshold", round(best_threshold, 4))
        mlflow.sklearn.log_model(model,  "isolation_forest")
        mlflow.sklearn.log_model(scaler, "scaler")

        run_id = mlflow.active_run().info.run_id

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(model,  MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    config = {
        "best_threshold" : round(best_threshold, 4),
        "contamination"  : contamination,
        "n_estimators"   : n_estimators,
        "window_short"   : 6,
        "window_long"    : 36,
        "roc_auc"        : round(roc_auc, 4),
        "f1_tuned"       : best_metrics["f1"],
        "run_id"         : run_id,
    }

    with open(MODELS_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[train] ✅ Done")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  F1        : {best_metrics['f1']:.4f}")
    print(f"  Threshold : {best_threshold:.4f}")
    print(f"  Run ID    : {run_id}")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument("--n-estimators",  type=int,   default=200)
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument("--run-name",      type=str,   default="isolation_forest")
    args = parser.parse_args()

    train(
        n_estimators  = args.n_estimators,
        contamination = args.contamination,
        run_name      = args.run_name,
    )