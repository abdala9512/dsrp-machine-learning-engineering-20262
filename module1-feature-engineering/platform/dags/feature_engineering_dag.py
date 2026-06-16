"""Module 1 — Feature Engineering pipeline orchestrated with Apache Airflow.

This DAG turns the manual steps from notebook 02 into a scheduled, observable
pipeline that CLOSES THE LOOP all the way to a trained, tracked model:

    extract -> prepare_feast_repo -> transform -> validate -> feast_apply
            -> feast_materialize -> train_model

  * extract            load raw Titanic data (synthetic fallback if offline)
  * prepare_feast_repo render a container-targeted Feast repo (redis service host)
  * transform          engineer features -> parquet (schema matches features.py)
  * validate           data-quality gate (schema / uniqueness / nulls)
  * feast_apply        register entities + feature views in the registry
  * feast_materialize  load latest feature values into the Redis online store
  * train_model        train a simple classifier and log it to MLflow
                       (MLFLOW_TRACKING_URI=http://mlflow:5000)

Run it from the Airflow UI at http://localhost:8080 (user/pass: airflow/airflow)
or trigger it from the CLI:  airflow dags trigger feature_engineering_pipeline
"""

from __future__ import annotations

import subprocess
from datetime import datetime

from airflow.decorators import dag, task

import feature_pipeline as fp

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 1,
}


def _run_feast(*args: str) -> str:
    """Run a feast CLI command inside the prepared repo and stream its output."""
    cmd = ["feast", "--chdir", str(fp.FEAST_REPO_DIR), *args]
    print(f"[feast] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"feast {' '.join(args)} failed (exit {result.returncode})")
    return result.stdout


@dag(
    dag_id="feature_engineering_pipeline",
    description="Engineer Titanic features and materialize them into Feast (Module 1).",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["module1", "feature-engineering", "feast", "mlflow"],
)
def feature_engineering_pipeline():
    @task
    def extract() -> str:
        return fp.extract()

    @task
    def prepare_feast_repo() -> str:
        return fp.prepare_feast_repo()

    @task
    def transform() -> str:
        return fp.build_features()

    @task
    def validate() -> dict:
        return fp.validate_features()

    @task
    def feast_apply() -> str:
        return _run_feast("apply")

    @task
    def feast_materialize() -> str:
        # materialize-incremental needs an end timestamp; "now" in ISO-8601.
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        return _run_feast("materialize-incremental", now)

    @task
    def train_model() -> dict:
        # Cierra el ciclo: training set desde el parquet del offline store,
        # entrena el modelo y lo registra en MLflow (http://mlflow:5000).
        return fp.train_model()

    raw = extract()
    repo = prepare_feast_repo()
    features = transform()
    checks = validate()
    applied = feast_apply()
    materialized = feast_materialize()
    trained = train_model()

    # Wiring: extract + repo prep both feed transform; then validate -> apply ->
    # materialize -> train_model (el entrenamiento corre DESPUES de materializar).
    [raw, repo] >> features >> checks >> applied >> materialized >> trained


dag = feature_engineering_pipeline()
