"""Hola mundo OFFLINE: features HISTORICAS (point-in-time) con Feast (Modulo 1).

Gemelo de `hello_feature_engineering` (que cierra el ciclo ONLINE hacia Redis).
Este DAG cierra el ciclo OFFLINE: construir un SET DE ENTRENAMIENTO correcto
con `get_historical_features`, la otra mitad de un feature store.

    create_dataset -> transform -+
                                 +-> feast_apply -> get_historical_features
    prepare_feast_repo ---------+

La pregunta que responde `get_historical_features` es:

    "¿Que valor tenia cada feature para ESTE usuario en ESTE momento?"

Le pasas un *entity dataframe* (pares user_id + event_timestamp, tipicamente el
momento en que ocurrio la etiqueta que quieres predecir) y Feast hace el JOIN
POINT-IN-TIME contra el offline store: para cada fila devuelve el ultimo valor
de la feature ANTERIOR a ese timestamp. Nunca un valor del futuro.

Eso es lo que evita el DATA LEAKAGE temporal: si entrenaras con el valor de HOY
de `clicks` para predecir algo que paso hace 5 dias, el modelo veria informacion
que no existia en ese momento (y en produccion se degradaria en silencio).

Como el dataset es ACUMULADO (clicks solo crece dia a dia), el efecto se ve a
simple vista en los logs: el mismo user_id pedido en 3 fechas distintas devuelve
3 valores distintos de clicks — cada uno el vigente en esa fecha.

El resultado se guarda como `hello_training_set.parquet` en /feast_data.

Ejecutar: Airflow UI en http://localhost:8080 (admin/airflow), des-pausar
`hello_historical_features` y darle ▶ Trigger. Revisa los logs de la tarea
`get_historical_features`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from airflow.decorators import dag, task

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hello_pipeline as hp  # noqa: E402

# Script que corre con el python del venv de feast (el SDK vive alli).
# Pide el MISMO usuario en VARIOS momentos del pasado -> valores distintos,
# cada uno el vigente en ese momento (join point-in-time).
GET_HISTORICAL_PY = f'''\
from datetime import timedelta

import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# El "hoy" del dataset = el ultimo timestamp disponible en el offline store.
hist = pd.read_parquet("{hp.FEATURES_PATH}")
latest = hist["event_timestamp"].max()

# Entity dataframe: QUE usuarios y EN QUE MOMENTO. En un caso real, cada fila
# seria "usuario X en el momento en que ocurrio su etiqueta" (compro / no compro).
entity_df = pd.DataFrame(
    [
        # usuario 1 en tres momentos distintos -> tres versiones de sus features
        {{"user_id": 1, "event_timestamp": latest - timedelta(days=5)}},
        {{"user_id": 1, "event_timestamp": latest - timedelta(days=2)}},
        {{"user_id": 1, "event_timestamp": latest}},
        # el resto, en momentos variados
        {{"user_id": 2, "event_timestamp": latest - timedelta(days=3)}},
        {{"user_id": 3, "event_timestamp": latest - timedelta(days=1)}},
        {{"user_id": 4, "event_timestamp": latest - timedelta(days=6)}},
        {{"user_id": 5, "event_timestamp": latest}},
    ]
)

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_stats:clicks",
        "user_stats:purchases",
        "user_stats:conversion_rate",
    ],
).to_df().sort_values(["user_id", "event_timestamp"]).reset_index(drop=True)

print("set de entrenamiento point-in-time (get_historical_features):")
print(training_df.to_string(index=False))

# Prueba de que el join es point-in-time: el usuario 1, pedido en 3 fechas,
# devuelve clicks CRECIENTES (el dataset es acumulado) y ninguno "del futuro".
u1 = training_df[training_df["user_id"] == 1]["clicks"].tolist()
assert u1 == sorted(u1) and u1[0] < u1[-1], f"point-in-time roto: {{u1}}"
print(f"OK point-in-time: user_id=1 tenia clicks={{u1[0]}} hace 5 dias, "
      f"{{u1[1]}} hace 2 dias y {{u1[2]}} hoy — nunca un valor del futuro.")

training_df.to_parquet("{hp.TRAINING_SET_PATH}", index=False)
print("training set -> {hp.TRAINING_SET_PATH}")
'''


@dag(
    dag_id="hello_historical_features",
    description="Hola mundo OFFLINE: set de entrenamiento point-in-time con get_historical_features.",
    schedule=None,  # solo trigger manual; es una demo
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"owner": "ml-platform", "retries": 0},
    tags=["module1", "hello-world", "feast", "point-in-time"],
)
def hello_historical_features():
    @task
    def create_dataset() -> str:
        return hp.create_dataset()

    @task
    def transform() -> str:
        return hp.build_features()

    @task
    def prepare_feast_repo() -> str:
        return hp.prepare_feast_repo()

    @task
    def feast_apply() -> str:
        return hp.run_feast("apply")

    @task
    def get_historical_features() -> str:
        """Join point-in-time contra el offline store -> training set en parquet."""
        return hp.run_feast_python("get_historical.py", GET_HISTORICAL_PY)

    features = transform()
    repo = prepare_feast_repo()

    create_dataset() >> features
    [features, repo] >> feast_apply() >> get_historical_features()


dag = hello_historical_features()
