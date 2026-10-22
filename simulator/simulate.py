import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_URL        = "http://localhost:8000/predict"
DATA_PATH      = Path(__file__).parent.parent / "data" / "nab_aws_merged.csv"
WINDOW_SIZE    = 50     # rows sent per request (needs min 37)
STEP_SIZE      = 5      # rows to advance each tick
INTERVAL_SEC   = 5      # seconds between each tick
INJECT_ANOMALY = True   # randomly inject extra anomalies for demo


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df = df[["cpu_util_1", "cpu_util_2", "rds_cpu"]].copy()
    print(f"[simulator] Loaded {len(df)} rows from {DATA_PATH.name}")
    return df


def inject_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Randomly spike one metric to simulate a live anomaly."""
    df = df.copy()
    spike_col = np.random.choice(["cpu_util_1", "cpu_util_2", "rds_cpu"])
    spike_idx = np.random.randint(len(df) - 5, len(df))  # spike near end
    if spike_col == "cpu_util_1":
        df.iloc[spike_idx, df.columns.get_loc(spike_col)] *= 8
    elif spike_col == "cpu_util_2":
        df.iloc[spike_idx, df.columns.get_loc(spike_col)] *= 12
    else:
        df.iloc[spike_idx, df.columns.get_loc(spike_col)] *= 3
    print(f"[simulator] ⚡ Injected spike in {spike_col} at row {spike_idx}")
    return df


def send_window(window: pd.DataFrame) -> dict | None:
    """Send a window of rows to the API and return the response."""
    records = window.reset_index()
    records["timestamp"] = records["timestamp"].astype(str)
    payload = {"records": records.to_dict(orient="records")}

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[simulator] API error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("[simulator] ❌ Cannot connect to API — is it running?")
        return None
    except Exception as e:
        print(f"[simulator] Error: {e}")
        return None


def print_result(result: dict, tick: int):
    """Pretty print the prediction result."""
    total    = result["total_records"]
    anomalies= result["anomaly_count"]
    rate     = result["anomaly_rate_pct"]
    threshold= result["threshold_used"]

    status = "🔴 ANOMALY" if anomalies > 0 else "✅ NORMAL"

    print(f"\n{'='*60}")
    print(f"Tick {tick:>4} | {datetime.now().strftime('%H:%M:%S')} | {status}")
    print(f"{'='*60}")
    print(f"  Records scored : {total}")
    print(f"  Anomalies      : {anomalies} ({rate}%)")
    print(f"  Threshold      : {threshold}")

    if anomalies > 0:
        print(f"\n  Anomalous timestamps:")
        for p in result["predictions"]:
            if p["is_anomaly"] == 1:
                print(f"    → {p['timestamp']} | "
                      f"score: {p['anomaly_score']} | "
                      f"confidence: {p['confidence']}")


def run():
    print("="*60)
    print("  Anomaly Detection Simulator")
    print(f"  API     : {API_URL}")
    print(f"  Window  : {WINDOW_SIZE} rows")
    print(f"  Step    : {STEP_SIZE} rows")
    print(f"  Interval: {INTERVAL_SEC}s")
    print("="*60)

    df = load_data()
    n_rows = len(df)
    tick = 0
    pos  = 0

    # Inject anomaly every N ticks for demo purposes
    inject_every = 6

    while True:
        # Wrap around when we reach the end
        if pos + WINDOW_SIZE > n_rows:
            pos = 0
            print("\n[simulator] 🔄 Reached end of data — restarting from beginning")

        # Get current window
        window = df.iloc[pos : pos + WINDOW_SIZE].copy()

        # Inject anomaly periodically for a good demo
        if INJECT_ANOMALY and tick % inject_every == 0 and tick > 0:
            window = inject_anomaly(window)

        # Send to API
        result = send_window(window)

        if result:
            print_result(result, tick)

        # Advance
        pos  += STEP_SIZE
        tick += 1

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    run()