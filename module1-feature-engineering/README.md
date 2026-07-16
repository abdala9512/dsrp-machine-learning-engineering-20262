# Modulo 1 — Feature Engineering & Feature Stores

Curso de Advanced ML Engineering. Este modulo cubre la transformacion de datos
crudos en **features** listas para un modelo, como gestionarlas a escala con un
**feature store** (Feast), y como **cerrar el ciclo** hasta un modelo entrenado y
evaluado. Trabaja sobre el dataset **Ames Housing** (Kaggle "House Prices",
`data/housing_train.csv`, 1460 casas) y la tarea es de **regresion**: predecir
`SalePrice`. Ademas, este modulo **introduce la plataforma compartida del curso**:
un unico `docker-compose` que levanta Feast y Airflow.

> El seguimiento de experimentos y el registro de modelos con **MLflow** se
> introducen en el **Modulo 2** (entrenamiento de modelos), no aqui.

## Objetivos

Al terminar este modulo seras capaz de:

1. Aplicar el flujo central de feature engineering y saber *cuando* conviene cada
   tecnica: **seleccion de variables** (filter / wrapper / embedded),
   **imputacion** de faltantes, **encoding** (label/ordinal, one-hot, hashing),
   **transformaciones** (log, Box-Cox / Yeo-Johnson, binning), **escalado**
   (z-score, robusto, min-max, L2), reduccion de dimensionalidad y embeddings.
2. Razonar la matematica detras de cada transformacion (z-score, min-max, norma
   L2, descomposicion en autovalores de PCA, el hashing trick) con la intuicion
   por delante.
3. **Detectar y tratar outliers / anomalias** como parte de la limpieza de datos:
   metodos estadisticos (z-score, IQR/Tukey, MAD) y basados en ML (Isolation
   Forest, DBSCAN), y decidir entre eliminar, winsorizar o transformar.
4. Explicar que resuelve un feature store: **sesgo entrenamiento-serving**,
   **correctitud point-in-time** y **reutilizacion de features**.
5. Construir un pipeline de Feast de punta a punta: definir features, `apply`,
   `materialize` y recuperar features **historicas** (entrenamiento) y **online**
   (serving), respaldadas por Redis + Postgres en Docker.
6. **Cerrar el ciclo**: construir un set de entrenamiento desde el offline store,
   entrenar un modelo de **regresion** que predice `SalePrice` y evaluarlo
   (RMSE / MAE / R² y el RMSE sobre `log(SalePrice)`, la metrica de Kaggle). El
   *experiment tracking* con MLflow se introduce en el **Modulo 2**.
7. **Orquestar** todo el pipeline como un DAG programado y observable en
   **Apache Airflow** — el **orquestador compartido del curso**, introducido aqui.

## Contenido

```
module1-feature-engineering/
├── README.md                       <- estas aqui
├── pyproject.toml                  <- dependencias (gestionadas con uv)
├── uv.lock                         <- versiones fijadas (reproducible)
├── notebooks/
│   ├── assets/header.png                      <- imagen de cabecera (en cada notebook)
│   ├── 01_feature_engineering_theory.ipynb   <- seleccion, imputacion, encoding, transformaciones, escalado, PCA, embeddings
│   ├── 02_outlier_detection.ipynb            <- deteccion de outliers/anomalias (estadistica + ML)
│   ├── 03_feature_pipeline_feast.ipynb       <- feature store + Feast en la practica
│   └── 04_pipeline_entrenamiento.ipynb       <- feature store -> regresor (SalePrice) evaluado
├── platform/                       <- PLATAFORMA COMPARTIDA DEL CURSO
│   ├── docker-compose.yml          <- redis + feast-ui + airflow (ligero, 3 contenedores)
│   ├── Dockerfile.airflow          <- apache/airflow:2.10.5 + solo deps del Modulo 1
│   ├── Dockerfile.feast            <- imagen minima para la UI web de Feast
│   ├── requirements-airflow.txt    <- deps del DAG del Modulo 1 (ligero)
│   ├── README.md                   <- guia de la plataforma (Feast + Airflow)
│   ├── feature_repo/               <- repo de Feast (feature_store.yaml, features.py, data/)
│   └── dags/                       <- DAG del Modulo 1 (feature_engineering_dag.py + feature_pipeline.py)
└── data/                           <- datos de trabajo
```

> **Renombre importante:** las antiguas carpetas `feast/` y `airflow/` se fusionaron
> en **`platform/`**. El repo de Feast vive ahora en `platform/feature_repo/` y los
> DAGs en `platform/dags/`. Hay un unico `docker-compose.yml` para toda la
> infraestructura compartida.

> Todos los notebooks llevan una **imagen de cabecera** (`notebooks/assets/header.png`)
> en su primera celda, con titulo, subtitulo, modulo y profesor.

### Notebook 1 — Teoria de Feature Engineering

Trabaja sobre el dataset **Ames Housing** (`data/housing_train.csv`, con un helper
que localiza el CSV y un fallback sintetico sin conexion). Como el objetivo es
`SalePrice` (continuo), las tecnicas estan adaptadas a **regresion**. El flujo va de
la seleccion a los embeddings, y cada tecnica combina **intuicion + una formula
compacta** con un ejemplo ejecutable:

- **Seleccion de variables** (al principio): filter (`VarianceThreshold`,
  correlacion, `f_regression`, `mutual_info_regression` — **no `chi2`**, que es para
  objetivos categoricos), wrapper (`RFE` con un regresor) y embedded (**Lasso**,
  importancia de `RandomForestRegressor`) contra `SalePrice`.
- **Imputacion** de faltantes: Ames tiene faltantes REALES y con dos significados —
  `NaN` = "feature ausente" (`PoolQC`, `Alley`, `FireplaceQu`, `GarageType` ->
  `"None"`/0) vs faltante genuino (`LotFrontage` -> mediana, `Electrical` -> moda).
  `SimpleImputer`, `KNNImputer`, variable indicadora (`add_indicator`).
- **Encoding**: ordinal para las notas de calidad reales (`ExterQual`, `BsmtQual`,
  `KitchenQual`, `HeatingQC`, `FireplaceQu` con orden Po<Fa<TA<Gd<Ex), one-hot para
  nominales (`MSZoning`, `BldgType`, `Foundation`, `CentralAir`) y feature hashing
  del `Neighborhood` de alta cardinalidad (25 barrios; colisiones, memoria).
- **Transformaciones**: log / `log1p` de `SalePrice` (skew 1.88 -> 0.12; ¡es la
  metrica de Kaggle!) y `LotArea`/`GrLivArea`; Box-Cox y Yeo-Johnson
  (`PowerTransformer`) con histogramas antes/despues y skew; binning
  (`YearBuilt` -> decadas, `GrLivArea` -> cuantiles).
- **Escalado**: estandarizacion (z-score), robusto (mediana/IQR; ideal por los
  outliers de Ames), min-max y L2.
- **Reduccion de dimensionalidad** (PCA: covarianza, autovectores, varianza explicada).
- **Embeddings** del `Neighborhood` (vectores densos aprendidos vs one-hot;
  `nn.Embedding` de PyTorch, dim 8).

Cierra con una **tabla resumen** de cuando usar cada tecnica.

### Notebook 2 — Deteccion de Outliers y Anomalias

Trata los outliers / anomalias como un paso de **limpieza de datos** dentro del
feature engineering, usando el ejemplo canonico de Ames: el scatter de `GrLivArea`
vs `SalePrice` y las **4 casas con `GrLivArea > 4000`** (dos ventas parciales
vendidas baratas). Cubre **metodos estadisticos** (z-score, regla IQR / vallas de
Tukey, z-score modificado con MAD sobre `GrLivArea`, `LotArea`, `SalePrice`) con
boxplots y scatter, **metodos basados en ML** (Isolation Forest y DBSCAN sobre el
plano 2D `[GrLivArea, SalePrice]` estandarizado) con comparacion por ROC-AUC, y
**que hacer** con ellos (eliminar las 4 casas gigantes, winsorizar o transformar).
Sin MLflow.

### Notebook 3 — Pipeline de Features con Feast

Teoria de un feature store (offline vs online store, correctitud point-in-time,
materializacion, reutilizacion) seguida de un pipeline practico de Feast que hace
ingenieria de un conjunto curado de features de Ames Housing (entidad `house` con
clave `house_id`, nombres snake_case), las escribe a parquet y corre `apply` /
`materialize` / `get_historical_features` / `get_online_features`.

### Notebook 4 — Pipeline de Entrenamiento (cierra el ciclo)

Construye el set de entrenamiento desde el **offline store** de Feast con
`get_historical_features` (con fallback a parquet), entrena un **regresor**
(`RandomForestRegressor` sobre `log1p(SalePrice)`) que predice `SalePrice` y lo
evalua (RMSE / MAE / R² y el RMSE sobre `log(SalePrice)` de Kaggle), con un grafico
predicho vs real. Explica el concepto central: *feature store → set de
entrenamiento → modelo* y la **consistencia entrenamiento-serving**. El
*experiment tracking* y el Model Registry con MLflow se cubren en el **Modulo 2**.

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

## El pipeline cerrado con Airflow

Todo el ciclo corre programado en Airflow, **hasta el modelo entrenado y
evaluado**:

```
 DAG: feature_engineering_pipeline
 extract -> prepare_feast_repo -> transform -> validate -> feast_apply
         -> feast_materialize -> train_model
   crudo      repo Feast          features     DQ gate     registry
              (host redis)                                  -> Redis  -> modelo
```

`platform/docker-compose.yml` levanta, en una sola red, **3 contenedores ligeros**:

- **`redis`**: online store de Feast.
- **`feast-ui`**: UI web de Feast en `http://localhost:8888`.
- **`airflow`**: **Airflow 3 en un solo contenedor** (`airflow standalone`, SQLite)
  con la UI en `http://localhost:8080` (login **`admin`/`airflow`**). Por defecto
  monta solo el DAG del Modulo 1; ver `platform/README.md` para orquestar tambien
  otros modulos.

El Postgres de Feast es opcional (registry SQL) y no arranca salvo que uses el
perfil `feast-sql-registry`.

La tarea `train_model` lee el parquet del offline store, entrena un
`RandomForestRegressor` que predice `SalePrice` (sobre `log1p`), lo evalua
(RMSE / MAE / R²), deja las metricas en los logs de la tarea y guarda el modelo en
disco (joblib).

> El **seguimiento de experimentos** y el **registro de modelos** con MLflow se
> introducen en el **Modulo 2** (entrenamiento de modelos).

Detalles completos en [`platform/README.md`](platform/README.md).

## Prerrequisitos

- Python 3.10–3.12
- [**uv**](https://docs.astral.sh/uv/) para gestionar dependencias
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker + Docker Compose
- La plataforma es **ligera**: 3 contenedores (redis + feast-ui + airflow) y una
  imagen de Airflow pequena (solo deps del Modulo 1). Ver `platform/README.md`.

## Setup

Este modulo usa **uv** (no `requirements.txt`). Las dependencias estan en
[`pyproject.toml`](pyproject.toml) y fijadas en `uv.lock`.

```bash
cd module1-feature-engineering

# 1. crear el entorno e instalar todo desde el lockfile
uv sync

# 2. abrir JupyterLab y correr los notebooks 1 -> 2 -> 3 -> 4
uv run jupyter lab
```

> `uv sync` crea el entorno (`.venv`) y resuelve las dependencias de forma
> reproducible. Para correr cualquier comando dentro del entorno usa `uv run ...`.

## Correr el pipeline

**Manual** (notebook / CLI):

```bash
# arranca solo el online store de Feast (+ la UI de Feast, opcional)
cd platform && docker compose up -d redis feast-ui

# corre el notebook 03 (Feast) para generar
#   platform/feature_repo/data/housing_features.parquet
# luego, desde el repo de features:
cd feature_repo
feast apply
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```

**Orquestado** (Airflow corre cada paso, incluido el entrenamiento):

```bash
# construye + levanta toda la plataforma (Feast + Airflow)
cd platform && docker compose up -d --build
# abre http://localhost:8080 (login admin/airflow), des-pausa y dispara
#   "feature_engineering_pipeline"
# las metricas del modelo quedan en los logs de la tarea train_model
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
- **Experiment tracking:** este modulo *no* usa MLflow. El Notebook 4 y el DAG
  entrenan y evaluan el modelo y lo guardan en disco (joblib). El tracking de
  experimentos y el Model Registry con MLflow llegan en el **Modulo 2**.
- La plataforma esta configurada **ligera**: **Airflow 3** corre en **un solo
  contenedor** (`airflow standalone`, SQLite) y la imagen instala solo las deps del
  Modulo 1. `feast` va en un venv aislado (incompatibilidad de deps con Airflow 3).
  Para orquestar tambien otros modulos, ver `platform/README.md`.
- Los notebooks se distribuyen **sin** salidas ejecutadas; correlos de arriba a
  abajo.
