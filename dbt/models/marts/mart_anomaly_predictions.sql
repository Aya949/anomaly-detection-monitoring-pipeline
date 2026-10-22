-- Mart model: anomaly predictions with metric context
-- Joins predictions with original metrics for full context

SELECT
    p.id,
    p.timestamp,
    p.cpu_util_1,
    p.cpu_util_2,
    p.rds_cpu,
    p.anomaly_score,
    p.is_anomaly,
    p.confidence,
    p.threshold_used,
    p.model_version,
    p.predicted_at,

    -- Enrich with rolling context from features mart
    f.cpu_util_1_zscore,
    f.cpu_util_2_zscore,
    f.rds_cpu_zscore,
    f.cpu_util_2_mean_long,
    f.rds_cpu_mean_long,

    -- Human readable severity
    CASE
        WHEN p.anomaly_score > p.threshold_used * 1.2 THEN 'critical'
        WHEN p.anomaly_score > p.threshold_used * 1.05 THEN 'warning'
        WHEN p.anomaly_score > p.threshold_used THEN 'caution'
        ELSE 'normal'
    END AS severity

FROM marts.anomaly_predictions p
LEFT JOIN {{ ref('mart_metrics_features') }} f
    ON p.timestamp = f.metric_timestamp