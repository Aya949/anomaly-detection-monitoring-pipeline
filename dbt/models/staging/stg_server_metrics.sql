-- Staging model: clean and validate raw server metrics
-- Filters out nulls, validates ranges, casts types

WITH source AS (
    SELECT *
    FROM raw.server_metrics
),

cleaned AS (
    SELECT
        timestamp                                    AS metric_timestamp,
        ROUND(cpu_util_1::NUMERIC, 4)               AS cpu_util_1,
        ROUND(cpu_util_2::NUMERIC, 4)               AS cpu_util_2,
        ROUND(rds_cpu::NUMERIC, 4)                  AS rds_cpu,
        COALESCE(is_anomaly, 0)                     AS is_anomaly,
        loaded_at

    FROM source

    WHERE
        -- Remove nulls
        cpu_util_1 IS NOT NULL
        AND cpu_util_2 IS NOT NULL
        AND rds_cpu    IS NOT NULL

        -- Validate realistic ranges
        AND cpu_util_1 BETWEEN 0 AND 100
        AND cpu_util_2 BETWEEN 0 AND 100
        AND rds_cpu    BETWEEN 0 AND 100
),

deduplicated AS (
    SELECT DISTINCT ON (metric_timestamp) *
    FROM cleaned
    ORDER BY metric_timestamp, loaded_at DESC
)

SELECT * FROM deduplicated