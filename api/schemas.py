from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class MetricsInput(BaseModel):
    timestamp : str = Field(..., example="2014-02-20 10:00:00")
    cpu_util_1  : float = Field(..., ge=0,  example=0.13)
    cpu_util_2  : float = Field(..., ge=0,  example=5.2)
    rds_cpu     : float = Field(..., ge=0,  example=6.8)
    
class BatchInput(BaseModel):
    records : List[MetricsInput] = Field(
        ..., min_lenght =  37,
        description = "Minimum 37 rows needed for long rolling window"
    )
    
class PredictionResult(BaseModel):
    timestamp : str
    cpu_util_1    : float
    cpu_util_2    : float
    rds_cpu       : float
    anomaly_score : float
    is_anomaly    : int
    confidence    : str

class BatchPredictionResponse(BaseModel):
    total_records : int
    anomaly_count    : int
    anomaly_rate_pct : float
    threshold_used   : float
    predictions      : List[PredictionResult]
    
class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # fix pydantic warning
    
    status          : str
    model_loaded    : bool
    feature_count   : int
    threshold       : float
    roc_auc         : float