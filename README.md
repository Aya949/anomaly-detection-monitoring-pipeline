# Anomaly Detection & Monitoring Pipeline

An end-to-end data engineering and machine learning project for detecting anomalous server metrics, serving predictions through a REST API, orchestrating data workflows, and monitoring application performance.

The project combines Python, Scikit-learn, FastAPI, PostgreSQL, Apache Airflow, dbt, Docker, Prometheus, and Grafana to explore how anomaly detection can be integrated into a data pipeline and monitored through an API.

## Architecture

<img width="1536" height="764" alt="63ea6a55-3733-4dc4-be42-38e771174f47" src="https://github.com/user-attachments/assets/f49f58ab-59b5-48cc-a56d-ba9a2fbdc23e" />


The system brings together four main components:

- **Data engineering:** server-metric datasets, PostgreSQL storage, Airflow orchestration, and dbt transformations.
- **Machine learning:** feature engineering, Isolation Forest training, and model evaluation.
- **Model serving:** a FastAPI service for batch anomaly predictions.
- **Observability:** Prometheus metrics collection and Grafana dashboards.

## Key Features

### Machine Learning
- Exploratory data analysis and feature analysis.
- Time-series feature engineering.
- Isolation Forest for unsupervised anomaly detection.
- Feature scaling and persisted model artifacts.
- Model evaluation and prediction visualizations.

### Data Engineering
- PostgreSQL database initialization.
- Airflow DAG for pipeline orchestration.
- dbt staging and mart models.
- SQL-based data-quality tests.
- Docker-based service configuration.

### API & Monitoring
- FastAPI prediction endpoint.
- Input validation using Pydantic schemas.
- Prediction and anomaly counters.
- Inference-latency metrics.
- Prometheus scraping and Grafana visualization.
- API alerting components.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python, SQL |
| Machine learning | Scikit-learn, Isolation Forest |
| API | FastAPI, Pydantic |
| Data storage | PostgreSQL |
| Orchestration | Apache Airflow |
| Transformations | dbt |
| Containers | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana |
| Experiment tracking | MLflow |
| Testing | pytest |
| Analysis | Jupyter Notebook, Pandas, Matplotlib |

## Project Structure

```text
anomaly-detection-monitoring-pipeline/
├── airflow/
│   └── dags/
│       └── metrics_pipeline.py
├── api/
│   ├── __init__.py
│   ├── alerting.py
│   ├── main.py
│   ├── predictor.py
│   ├── schemas.py
│   ├── Dockerfile
│   └── requirements.txt
├── artifacts/
│   └── feature_cols.txt
├── data/
│   ├── ec2_cpu_utilization_24ae8d.csv
│   ├── ec2_cpu_utilization_fe7f93.csv
│   ├── ec2_network_in_5abac7.csv
│   ├── nab_aws_merged.csv
│   ├── nab_features.csv
│   └── rds_cpu_utilization_cc0c53.csv
├── db/
│   └── init.sql
├── dbt/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/
│   │   │   └── stg_server_metrics.sql
│   │   └── marts/
│   │       ├── mart_metrics_features.sql
│   │       └── mart_anomaly_predictions.sql
│   └── tests/
│       └── test_no_nulls.sql
├── docs/
│   └── screenshots/
├── grafana/
│   ├── dashboards/
│   │   └── anomaly_detection.json
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboard.yml
│       └── datasources/
│           └── prometheus.yml
├── models/
│   ├── feature_cols.json
│   ├── isolation_forest.pkl
│   ├── model_config.json
│   └── scaler.pkl
├── Notebooks/
│   ├── data.ipynb
│   ├── test_payload.json
│   ├── artifacts/
│   └── plots/
├── plots/
├── prometheus/
│   └── prometheus.yml
├── simulator/
│   ├── __init__.py
│   └── simulate.py
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── features.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   └── test_predictor.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Data Pipeline & Orchestration

Apache Airflow orchestrates the workflow defined in `airflow/dags/metrics_pipeline.py`.

The DAG is configured around these stages:

1. **Extract and load:** ingest metrics into PostgreSQL.
2. **Validate data:** check the data before downstream transformations.
3. **Run dbt:** execute warehouse transformations.
4. **Test dbt models:** run data-quality checks.
5. **Log pipeline execution:** record pipeline-run information.

### dbt models

| Model | Purpose |
|---|---|
| `stg_server_metrics` | Staging layer for server metrics |
| `mart_metrics_features` | Feature-oriented analytical model |
| `mart_anomaly_predictions` | Model for anomaly prediction data |

The SQL models separate staging and analytical transformations. The `test_no_nulls.sql` file provides an additional data-quality check.

<img width="1470" height="480" alt="image" src="https://github.com/user-attachments/assets/11ba503b-7f2d-4bce-ab12-c7f7fb53d244" />



## Machine Learning

The project uses an **Isolation Forest** model to identify observations that differ from normal server-metric behavior.

### Workflow

1. Explore raw server-metric datasets.
2. Engineer features from the metrics.
3. Scale features using a fitted scaler.
4. Train and evaluate the anomaly detection model.
5. Persist the model and feature configuration.
6. Load the artifacts for API inference.

Relevant files:

- `src/features.py` — feature engineering.
- `src/train.py` — model training.
- `src/evaluate.py` — model evaluation.
- `models/isolation_forest.pkl` — serialized model.
- `models/scaler.pkl` — serialized feature scaler.
- `models/feature_cols.json` — feature configuration.

The `Notebooks/data.ipynb` notebook and the plot directories support exploratory analysis and evaluation.

### Model evaluation

The current recorded evaluation includes:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7095 |
| Configured threshold | 0.4611 |
| Model features | 30 |

These values describe the current saved model configuration and evaluation. ROC-AUC measures ranking discrimination; it does not by itself establish a suitable operational alert threshold.

## API & Model Serving

FastAPI exposes the model through an HTTP API.

Relevant implementation files:

- `api/main.py` — application and routes.
- `api/predictor.py` — prediction logic.
- `api/schemas.py` — request and response schemas.
- `api/alerting.py` — alerting functionality.

The prediction endpoint accepts a batch of server metrics and returns prediction results and summary statistics.

### API documentation

When the application is running locally, open:

**http://localhost:8000/docs**

<img width="1847" height="881" alt="image" src="https://github.com/user-attachments/assets/f2207b72-0ddf-4e88-ad37-a0f1d861681b" />

### Example prediction response

A successful test request returned the following summary:

```json
{
  "total_records": 7,
  "anomaly_count": 7,
  "anomaly_rate_pct": 100.0,
  "threshold_used": 0.4611
}
```
<img width="1005" height="410" alt="image" src="https://github.com/user-attachments/assets/1f926e55-9b0e-428f-8d65-fcaac712e67d" />


## Monitoring & Observability

Prometheus scrapes metrics exposed by the API, and Grafana visualizes application and model-inference behavior.

### Available metrics

| Metric | Description |
|---|---|
| `records_scored_total` | Cumulative number of records scored |
| `anomalies_detected_total` | Cumulative number of anomalies detected |
| `anomaly_rate_percent` | Current anomaly rate percentage |
| `inference_latency_seconds` | Histogram of inference durations |
| `http_requests_total` | HTTP requests by handler, method, and status |
| `http_request_duration_seconds` | HTTP request-duration histogram |
| `process_resident_memory_bytes` | API process resident memory |

Counters represent cumulative activity since the API process started. The anomaly-rate gauge represents a current value, while histogram metrics support latency analysis.

### Grafana dashboard

The dashboard is provisioned through the configuration files under `grafana/provisioning/`, with dashboard definitions stored in `grafana/dashboards/`.

<img width="1906" height="793" alt="image" src="https://github.com/user-attachments/assets/3c793fa0-70f5-49de-ae10-5f993a537b19" />


The dashboard can be used to inspect:

- API availability and request activity.
- Number of records scored and anomalies detected.
- Anomaly rate.
- Average and percentile latency.
- API process memory.

## Testing

The repository contains pytest tests for feature engineering and prediction logic:

```text
tests/
├── test_features.py
└── test_predictor.py
```

Run the tests from the project root after installing the project dependencies:

```bash
pytest
```

The project also includes a dbt data-quality test under `dbt/tests/`.

## Getting Started

### Prerequisites

- Git
- Docker
- Docker Compose
- A compatible Python environment for running local scripts and tests

### 1. Clone the repository

```bash
git clone https://github.com/Aya949/anomaly-detection-monitoring-pipeline.git
cd anomaly-detection-monitoring-pipeline
```

### 2. Configure environment variables

Create a local `.env` file using the supplied example:

```bash
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

Review `.env.example` and provide the values required by your local configuration. Do not commit real passwords or credentials.

### 3. Start the Docker services

```bash
docker compose up -d --build
```

Check service status:

```bash
docker compose ps
```

The exact service names, exposed ports, and startup requirements are defined in `docker-compose.yml`.

### 4. Access the application

If the corresponding services are running and their ports are exposed as configured:

| Component | Local address |
|---|---|
| FastAPI | http://localhost:8000 |
| FastAPI Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Use the credentials configured for your local Grafana instance. Do not publish secrets in this README.

### 5. Test the API

Use the Swagger interface at `http://localhost:8000/docs` to inspect the request schema and send a test prediction request.

A sample payload is available in `test_batch.json`. Ensure that the input matches the API schema before submitting it.

> **Note:** The Docker Compose workflow, dbt profile configuration, and service dependencies should be verified against the current repository before treating these instructions as a reproducible clean-machine installation guide.

## Limitations & Future Improvements

Potential improvements include:

- Add automated integration tests for the full pipeline.
- Validate the pipeline from a clean environment.
- Improve anomaly-threshold calibration using representative validation data.
- Add operational alert rules and alert-delivery tests.
- Monitor model quality and data drift over time.
- Verify and automate persistence of anomaly predictions in the analytical mart.
- Add a CI workflow to run tests and enforce code quality.

## Project Goals

This project explores the integration of data engineering, unsupervised anomaly detection, API deployment, workflow orchestration, and observability in one system.

It is intended as a practical demonstration of the engineering components surrounding a machine learning model—not just the model training step.

## Author

**Aya AIT ALLAL**
Data Engineer

- GitHub: [Aya949](https://github.com/Aya949)
- Project repository: [Anomaly Detection & Monitoring Pipeline](https://github.com/Aya949/anomaly-detection-monitoring-pipeline)
