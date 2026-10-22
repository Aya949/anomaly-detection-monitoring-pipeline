-- Mart model: pre-compute rolling aggregations for analysis
-- Window = 6 rows (30 min) and 36 rows (3 hours) at 5-min intervals

WITH base AS (
    SELECT *
    FROM {{ ref('stg_server_metrics') }}
    ORDER BY metric_timestamp
),

rolling AS (
    SELECT
        metric_timestamp,
        cpu_util_1,
        cpu_util_2,
        rds_cpu,
        is_anomaly,

        -- ── CPU util 1 ──────────────────────────────────────────────────────
        AVG(cpu_util_1) OVER (
            ORDER BY metric_timestamp
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS cpu_util_1_mean_short,

        STDDEV(cpu_util_1) OVER (
            ORDER BY metric_timestamp
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS cpu_util_1_std_short,

        AVG(cpu_util_1) OVER (
            ORDER BY metric_timestamp
            ROWS BETWEEN 35 PRECEDING AND CURRENT ROW
        ) AS cpu_util_1_mean_long,

        -- ── CPU util 2 ──────────────────────────────────────────────────────
        AVG(cpu_util_2) OVER (
            ORDER BY metric_timestamp
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS cpu_util_2_mean_short,

        STDDEV(cpu_util_2) OVER (
            ORDER BY metric_timestamp
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS cpu_util_2_std_short,

        AVG(cpu_util_2) OVER (
            ORDER BY metric_timestamp
            ROWS BETWEEN 35 PRECEDING AND CURRENT ROW
        ) AS cpu_util_2_mean_long,

        -- ── RDS CPU ─────────────────────────────────────────────────────────
        AVG(rds_cpu) OVER (
            ORDER BY metric_timestamp
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS rds_cpu_mean_short,

        STDDEV(rds_cpu) OVER (
            ORDER BY metric_timestamp
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS rds_cpu_std_short,

        AVG(rds_cpu) OVER (
            ORDER BY metric_timestamp
            ROWS BETWEEN 35 PRECEDING AND CURRENT ROW
        ) AS rds_cpu_mean_long

    FROM base
)

SELECT
    *,
    -- Z-scores
    CASE
        WHEN cpu_util_1_std_short > 0
        THEN (cpu_util_1 - cpu_util_1_mean_long) / cpu_util_1_std_short
        ELSE 0
    END AS cpu_util_1_zscore,

    CASE
        WHEN cpu_util_2_std_short > 0
        THEN (cpu_util_2 - cpu_util_2_mean_long) / cpu_util_2_std_short
        ELSE 0
    END AS cpu_util_2_zscore,

    CASE
        WHEN rds_cpu_std_short > 0
        THEN (rds_cpu - rds_cpu_mean_long) / rds_cpu_std_short
        ELSE 0
    END AS rds_cpu_zscore

FROM rolling