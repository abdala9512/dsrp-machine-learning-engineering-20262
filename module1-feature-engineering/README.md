# Modulo 1 — Feature Engineering & Feature Stores

Curso de Advanced ML Engineering. Este modulo cubre la transformacion de datos
crudos en **features** listas para un modelo, como gestionarlas a escala con un
**feature store** (Feast), y como **cerrar el ciclo** hasta un modelo entrenado y
registrado. Ademas, este modulo **introduce la plataforma compartida del curso**:
un unico `docker-compose` que levanta Feast, MLflow y Airflow para los cuatro
modulos.

## Objetivos

Al terminar este modulo seras capaz de:

1. Aplicar las transformaciones centrales de feature engineering y saber *cuando*
   conviene cada una: encoding, hashing, binning, escalado, normalizacion,
   reduccion de dimensionalidad y embeddings.
2. Razonar la matematica detras de cada transformacion (z-score, min-max, norma
   L2, descomposicion en autovalores de PCA, el hashing trick) con la intuicion
   por delante.
3. Explicar que resuelve un feature store: **sesgo entrenamiento-serving**,
   **correctitud point-in-time** y **reutilizacion de features**.
4. Construir un pipeline de Feast de punta a punta: definir features, `apply`,
   `materialize` y recuperar features **historicas** (entrenamiento) y **online**
   (serving), respaldadas por Redis + Postgres en Docker.
5. **Cerrar el ciclo**: construir un set de entrenamiento desde el offline store,
   entrenar un modelo, evaluarlo y registrarlo en **MLflow 3.x**.
6. **Orquestar** todo el pipeline como un DAG programado y observable en
   **Apache Airflow** — el **orquestador compartido del curso**, introducido aqui.

## Contenido

```
module1-feature-engineering/
├── README.md                       <- estas aqui
├── pyproject.toml                  <- dependencias (gestionadas con uv)
├── uv.lock                         <- versiones fijadas (reproducible)
├── notebooks/
│   ├── 01_feature_engineering_theory.ipynb   <- transformaciones + math + Titanic
│   ├── 02_feature_pipeline_feast.ipynb       <- feature store + Feast en la practica
│   └── 03_pipeline_entrenamiento.ipynb       <- feature store -> modelo -> MLflow
├── platform/                       <- PLATAFORMA COMPARTIDA DEL CURSO
│   ├── docker-compose.yml          <- redis + postgres (Feast) + MLflow + Airflow
│   ├── Dockerfile.airflow          <- apache/airflow:2.10.5-python3.11 + deps de los 4 modulos
│   ├── Dockerfile.mlflow           <- python:3.11-slim + mlflow>=3.1
│   ├── requirements-airflow.txt    <- deps combinadas de los DAGs de todos los modulos
│   ├── README.md                   <- guia de la plataforma (Feast + MLflow + Airflow)
│   ├── feature_repo/               <- repo de Feast (feature_store.yaml, features.py, data/)
│   └── dags/                       <- DAG del Modulo 1 (feature_engineering_dag.py + feature_pipeline.py)
└── data/                           <- datos de trabajo
```

> **Renombre importante:** las antiguas carpetas `feast/` y `airflow/` se fusionaron
> en **`platform/`**. El repo de Feast vive ahora en `platform/feature_repo/` y los
> DAGs en `platform/dags/`. Hay un unico `docker-compose.yml` para toda la
> infraestructura compartida.

### Notebook 1 — Teoria de Feature Engineering

Trabaja sobre el dataset del **Titanic** (`seaborn.load_dataset("titanic")`, con
fallback sintetico sin conexion). Cada tecnica combina **intuicion + una formula
compacta** con un ejemplo ejecutable:

- One-hot encoding (cardinalidad y dispersion)
- Feature hashing (el hashing trick, colisiones, trade-off de memoria)
- Bucketing / binning (uniforme vs cuantiles)
- Estandarizacion (z-score)
- Normalizacion (min-max y L2)
- Reduccion de dimensionalidad (PCA: covarianza, autovectores, varianza explicada)
- Embeddings (vectores densos aprendidos vs one-hot; `nn.Embedding` de PyTorch)

Cierra con una tabla-chuleta.

### Notebook 2 — Pipeline de Features con Feast

Teoria de un feature store (offline vs online store, correctitud point-in-time,
materializacion, reutilizacion) seguida de un pipeline practico de Feast que hace
ingenieria de features del Titanic, las escribe a parquet y corre `apply` /
`materialize` / `get_historical_features` / `get_online_features`.

### Notebook 3 — Pipeline de Entrenamiento (cierra el ciclo)

Construye el set de entrenamiento desde el **offline store** de Feast con
`get_historical_features` (con fallback a parquet), entrena un clasificador simple
que predice `survived`, lo evalua (accuracy / ROC-AUC) y lo registra en
**MLflow**. Explica el concepto central: *feature store → set de entrenamiento →
modelo* y la **consistencia entrenamiento-serving**.

## Arquitectura de Feast

```
   datos crudos -> feature engineering -> parquet (OFFLINE store, historia completa)
                                               |
                                     feast materialize
                                               v
                                  Redis (ONLINE store, ultimo valor/entidad)
                                               |
        get_historical_features <--------------+--------> get_online_features
        (entrenamiento, point-in-time)         |          (serving, ms latencia)
                                               v
                       REGISTRY (archivo registry.db o Postgres)
                       catalogo de entidades / feature views / fuentes
```

## El pipeline cerrado con Airflow + MLflow

Todo el ciclo corre programado en Airflow, **hasta el modelo entrenado y
registrado**:

```
 DAG: feature_engineering_pipeline
 extract -> prepare_feast_repo -> transform -> validate -> feast_apply
         -> feast_materialize -> train_model
   crudo      repo Feast          features     DQ gate     registry
              (host redis)                                  -> Redis    -> MLflow
```

`platform/docker-compose.yml` levanta, en una sola red:

- **Feast**: `redis` (online store) + `postgres` (registry/offline).
- **MLflow 3.x**: servidor en `http://localhost:5000` (en la red: `http://mlflow:5000`)
  + su propio Postgres de metadatos.
- **Airflow**: webserver (`http://localhost:8080`, `airflow`/`airflow`), scheduler
  y base de metadatos. Es el **orquestador compartido del curso**: monta los DAGs
  de los cuatro modulos en subcarpetas de `/opt/airflow/dags`.

La tarea `train_model` lee el parquet del offline store, entrena una
`LogisticRegression` que predice `survived`, evalua y registra params/metricas/modelo
en MLflow (`MLFLOW_TRACKING_URI=http://mlflow:5000`).

Detalles completos en [`platform/README.md`](platform/README.md).

## Prerrequisitos

- Python 3.10–3.12
- [**uv**](https://docs.astral.sh/uv/) para gestionar dependencias
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker + Docker Compose
- ~8 GB de disco libre (la imagen de Airflow es **pesada**: incluye deps de los 4
  modulos — sentence-transformers, xgboost, etc.; ver `platform/README.md`)

## Setup

Este modulo usa **uv** (no `requirements.txt`). Las dependencias estan en
[`pyproject.toml`](pyproject.toml) y fijadas en `uv.lock`.

```bash
cd module1-feature-engineering

# 1. crear el entorno e instalar todo desde el lockfile
uv sync

# 2. abrir JupyterLab y correr los notebooks 1 -> 2 -> 3
uv run jupyter lab
```

> `uv sync` crea el entorno (`.venv`) y resuelve las dependencias de forma
> reproducible. Para correr cualquier comando dentro del entorno usa `uv run ...`.

## Correr el pipeline

**Manual** (notebook / CLI):

```bash
# arranca solo los servicios de Feast (Redis + Postgres)
cd platform && docker compose up -d redis postgres

# corre el notebook 02 para generar
#   platform/feature_repo/data/titanic_features.parquet
# luego, desde el repo de features:
cd feature_repo
feast apply
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```

**Orquestado** (Airflow corre cada paso, incluido el entrenamiento):

```bash
# construye + levanta toda la plataforma (Feast + MLflow + Airflow)
cd platform && docker compose up -d --build
# abre http://localhost:8080 (airflow/airflow), des-pausa y dispara
#   "feature_engineering_pipeline"
# luego abre http://localhost:5000 para ver el modelo registrado en MLflow
```

Ver [`platform/README.md`](platform/README.md) para la referencia completa de
comandos y el teardown.

## Notas y caveats

- Feast cambia rapido. Estos materiales apuntan a **feast >= 0.40** (la API
  `Entity` / `FeatureView` / `FileSource` / `Field` + `feast.types`). Si instalas
  una version mucho mas vieja, los imports de `features.py` pueden diferir.
- El online store de Redis requiere el extra **`feast[redis]`** (ya en
  `pyproject.toml`). Un `feast` pelado falla en `feast apply` con
  `Could not import module 'feast.infra.online_stores.redis'`.
- El registry por defecto es un archivo local (`data/registry.db`); el registry de
  Postgres viene comentado como opcion en `feature_store.yaml`.
- **MLflow 3.x**: el servidor se construye desde `python:3.11-slim`
  (`mlflow>=3.1`) y se fija a la ultima 3.x disponible. Si el servidor no esta
  arriba, el Notebook 3 cae a un backend local en `./mlruns`.
- La imagen de Airflow es **grande** (deps de los 4 modulos). La primera build
  tarda varios minutos.
- Los notebooks se distribuyen **sin** salidas ejecutadas; correlos de arriba a
  abajo.
