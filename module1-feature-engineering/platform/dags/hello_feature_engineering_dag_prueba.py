"""Hola mundo ONLINE: feature engineering en Airflow conectado a Feast (Modulo 1).

Version MINIMA del patron que usa el pipeline real de Ames Housing
(`feature_engineering_pipeline`): un dataset sintetico de 5 usuarios x 7 dias,
UNA feature derivada, y el ciclo offline -> online completo:

    create_dataset -> transform -+
                                 +-> feast_apply -> feast_materialize -> read_online_features
    prepare_feast_repo ---------+

  * create_dataset        5 usuarios x 7 dias de clicks/purchases acumulados
  * transform             ingenieria de features: conversion_rate = purchases/clicks
  * prepare_feast_repo    renderiza el repo de Feast (hello_feature_repo/) para Docker
  * feast_apply           registra la entidad `user` y el feature view `user_stats`
  * feast_materialize     carga el ULTIMO valor de cada usuario a Redis (online store)
  * read_online_features  lee las features desde Redis via get_online_features()
                          — la misma llamada que haria un servicio de inferencia

Donde VERLO despues de correrlo:

  * UI de Feast: http://localhost:8888 — proyecto `hello_features` en el
    selector de proyectos (el registry compartido vive en /feast_data)
  * claves en Redis: RedisInsight en http://localhost:5540 (patron *hello_features)
  * logs de la tarea `read_online_features`: las features servidas desde Redis

El gemelo OFFLINE de este DAG es `hello_historical_features`: mismo dataset,
pero construye un set de entrenamiento point-in-time con get_historical_features.

Ejecutar: Airflow UI en http://localhost:8080 (admin/airflow), des-pausar
`hello_feature_engineering` y darle ▶ Trigger.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from airflow.decorators import dag, task

# El DAG y su modulo de logica (hello_pipeline.py) viven en la misma carpeta.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hello_pipeline as hp  # noqa: E402

# Script que corre con el python del venv de feast: lee features desde Redis
# con get_online_features() — exactamente lo que haria un servicio de inferencia.
READ_ONLINE_PY = '''\
from feast import FeatureStore

store = FeatureStore(repo_path=".")
result = store.get_online_features(
    features=[
        "user_stats:clicks",
        "user_stats:purchases",
        "user_stats:conversion_rate",
    ],
    entity_rows=[{"user_id": uid} for uid in (1, 2, 3, 4, 5)],
).to_dict()

print("features servidas desde Redis (online store) — el ULTIMO valor por usuario:")
for i, uid in enumerate(result["user_id"]):
    print(
        f"  user_id={uid}  clicks={result['clicks'][i]}  "
        f"purchases={result['purchases'][i]}  "
        f"conversion_rate={result['conversion_rate'][i]:.3f}"
    )

# Si Redis estuviera vacio (sin materialize) los valores serian None.
assert all(v is not None for v in result["conversion_rate"]), "online store vacio!"
print("OK: el ciclo offline -> online quedo cerrado.")
'''


@dag(
    dag_id="prueba_dsrp",
    description="Hola mundo ONLINE: Airflow + Feast + Redis (dataset sintetico de usuarios).",
    schedule="0 8 * * 1",  # solo trigger manual; es una demo
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"owner": "ml-platform", "retries": 0},
    tags=["module1", "hello-world", "feast"],
)
def hello_feature_engineering():
    @task
    def crear_dataset() -> str:
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
    def feast_materialize() -> str:
        """Carga el ultimo valor por usuario del parquet (offline) a Redis (online)."""
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        return hp.run_feast("materialize-incremental", now)

    @task
    def read_online_features() -> str:
        """Lee las features desde Redis, como lo haria un servicio de inferencia."""
        return hp.run_feast_python("read_online.py", READ_ONLINE_PY)

    features = transform()
    repo = prepare_feast_repo()
    _feast_apply = feast_apply()

    # El apply necesita el repo renderizado Y el parquet en su sitio.
    crear_dataset() >> features
    [features, repo] >> _feast_apply >> feast_materialize() >> read_online_features()


dag = hello_feature_engineering()
