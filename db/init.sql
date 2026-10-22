-- ── Schemas ───────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS marts;

-- ── Raw table — exact copy of CSV data ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw.server_metrics (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMP NOT NULL,
    cpu_util_1   FLOAT,
    cpu_util_2   FLOAT,
    rds_cpu      FLOAT,
    is_anomaly   INTEGER DEFAULT 0,
    loaded_at    TIMESTAMP DEFAULT NOW()
);

-- Prevent duplicate timestamps
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_timestamp
    ON raw.server_metrics(timestamp);

-- ── Predictions table — stores all API predictions ────────────────────────────
CREATE TABLE IF NOT EXISTS marts.anomaly_predictions (
    id             SERIAL PRIMARY KEY,
    timestamp      TIMESTAMP NOT NULL,
    cpu_util_1     FLOAT,
    cpu_util_2     FLOAT,
    rds_cpu        FLOAT,
    anomaly_score  FLOAT,
    is_anomaly     INTEGER,
    confidence     VARCHAR(10),
    threshold_used FLOAT,
    model_version  VARCHAR(50) DEFAULT 'isolation_forest_v1',
    predicted_at   TIMESTAMP DEFAULT NOW()
);

-- ── Pipeline runs — tracks each DAG execution ─────────────────────────────────
CREATE TABLE IF NOT EXISTS marts.pipeline_runs (
    id           SERIAL PRIMARY KEY,
    run_id       VARCHAR(100),
    status       VARCHAR(20),
    rows_loaded  INTEGER,
    started_at   TIMESTAMP DEFAULT NOW(),
    finished_at  TIMESTAMP
);