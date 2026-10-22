-- Test: no null timestamps in staging
SELECT COUNT(*)
FROM {{ ref('stg_server_metrics') }}
WHERE metric_timestamp IS NULL
HAVING COUNT(*) > 0