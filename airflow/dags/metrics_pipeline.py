"""
Airflow DAG: Server Metrics Pipeline
Runs hourly — loads raw data → runs dbt → validates quality
"""
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import psycopg2
import subprocess
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host"    : "postgres",
    "port"    : 5432,
    "dbname"  : "anomaly_db",
    "user"    : "airflow",
    "password": os.environ["POSTGRES_PASSWORD"],
}

DATA_PATH = "/opt/airflow/data/nab_aws_merged.csv"

default_args = {
    "owner"           : "airflow",
    "retries"         : 2,
    "retry_delay"     : timedelta(minutes=2),
    "email_on_failure": False,
}


# ── Tasks ─────────────────────────────────────────────────────────────────────
def extract_and_load(**context):
    """
    Extract data from CSV and load into PostgreSQL raw schema.
    Uses upsert to avoid duplicates.
    """
    print(f"[extract] Loading data from {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df[["timestamp", "cpu_util_1", "cpu_util_2",
             "rds_cpu", "is_anomaly"]].copy()
    df = df.dropna()

    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    inserted = 0
    skipped  = 0

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO raw.server_metrics
                    (timestamp, cpu_util_1, cpu_util_2, rds_cpu, is_anomaly)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (timestamp) DO NOTHING
            """, (
                row["timestamp"],
                float(row["cpu_util_1"]),
                float(row["cpu_util_2"]),
                float(row["rds_cpu"]),
                int(row["is_anomaly"]),
            ))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[extract] Error on row {row['timestamp']}: {e}")
            conn.rollback()
            continue

    conn.commit()
    cursor.close()
    conn.close()

    print(f"[extract] ✅ Inserted: {inserted} | Skipped: {skipped}")

    # Push to XCom for downstream tasks
    context["ti"].xcom_push(key="rows_inserted", value=inserted)
    return inserted


def validate_data(**context):
    """
    Basic data quality checks on raw table.
    Fails the task if quality issues found.
    """
    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    issues = []

    # Check 1 — row count
    cursor.execute("SELECT COUNT(*) FROM raw.server_metrics")
    count = cursor.fetchone()[0]
    print(f"[validate] Total rows in raw: {count}")
    if count == 0:
        issues.append("Raw table is empty")

    # Check 2 — no nulls in key columns
    cursor.execute("""
        SELECT COUNT(*) FROM raw.server_metrics
        WHERE cpu_util_1 IS NULL
           OR cpu_util_2 IS NULL
           OR rds_cpu IS NULL
    """)
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        issues.append(f"Found {null_count} rows with null values")

    # Check 3 — values in realistic range
    cursor.execute("""
        SELECT COUNT(*) FROM raw.server_metrics
        WHERE cpu_util_1 < 0 OR cpu_util_1 > 100
           OR cpu_util_2 < 0 OR cpu_util_2 > 100
           OR rds_cpu    < 0 OR rds_cpu    > 100
    """)
    out_of_range = cursor.fetchone()[0]
    if out_of_range > 0:
        issues.append(f"Found {out_of_range} rows with out-of-range values")

    # Check 4 — anomaly rate sanity check
    cursor.execute("""
        SELECT
            ROUND(AVG(is_anomaly::FLOAT)::NUMERIC * 100, 2)
        FROM raw.server_metrics
    """)
    anomaly_rate = cursor.fetchone()[0]
    print(f"[validate] Anomaly rate: {anomaly_rate}%")

    cursor.close()
    conn.close()

    if issues:
        raise ValueError(f"Data quality issues found: {issues}")

    print("[validate] ✅ All quality checks passed")
    return True


def log_pipeline_run(**context):
    """Log pipeline run metadata to PostgreSQL."""
    rows_inserted = context["ti"].xcom_pull(
        key="rows_inserted", task_ids="extract_and_load"
    )

    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO marts.pipeline_runs
            (run_id, status, rows_loaded, finished_at)
        VALUES (%s, %s, %s, NOW())
    """, (
        context["run_id"],
        "success",
        rows_inserted or 0,
    ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[log] ✅ Pipeline run logged")


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id          = "server_metrics_pipeline",
    description     = "ETL pipeline for server metrics anomaly detection",
    default_args    = default_args,
    start_date      = days_ago(1),
    schedule_interval= "@hourly",
    catchup         = False,
    tags            = ["anomaly-detection", "etl", "mlops"],
) as dag:

    extract_load = PythonOperator(
        task_id         = "extract_and_load",
        python_callable = extract_and_load,
    )

    validate = PythonOperator(
        task_id         = "validate_data",
        python_callable = validate_data,
    )

    dbt_run = BashOperator(
        task_id      = "dbt_run",
        bash_command = "cd /opt/airflow/dbt && dbt run --profiles-dir .",
    )

    dbt_test = BashOperator(
        task_id      = "dbt_test",
        bash_command = "cd /opt/airflow/dbt && dbt test --profiles-dir .",
    )

    log_run = PythonOperator(
        task_id         = "log_pipeline_run",
        python_callable = log_pipeline_run,
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    extract_load >> validate >> dbt_run >> dbt_test >> log_run
