# Módulo 2 — Algoritmos avanzados de ML (+ MLflow)

Un recorrido práctico por los algoritmos que todo ingeniero de ML debería
dominar, acompañado de **tracking de experimentos y un model registry** con
**MLflow 3.x**. Cada notebook combina *teoría + matemática* con *código
ejecutable* sobre datasets *reales y offline* de scikit-learn, y registra su
trabajo en MLflow.

## Objetivos

Al terminar este módulo serás capaz de:

- Aplicar **regularización** (Ridge / Lasso / ElasticNet) y razonar sobre el
  compromiso sesgo–varianza.
- Construir y comparar **ensembles** — Random Forest y las tres librerías
  modernas de boosting (**XGBoost, LightGBM, CatBoost**) — y explicar *en qué se
  diferencian*.
- Entrenar y ajustar **máquinas de soporte vectorial (SVM)** y entender el truco
  del kernel.
- Combinar modelos con **votación** (dura / suave) y **stacking / blending**
  (meta-aprendizaje out-of-fold).
- Ejecutar pipelines **no supervisados**: DBSCAN/OPTICS, KMeans/GMM y t-SNE/UMAP
  para visualización.
- Implementar una **red neuronal en PyTorch** de principio a fin (la neurona, las
  activaciones, el desvanecimiento del gradiente, backprop, SGD/optimizadores,
  regularización) con un bucle de entrenamiento completo, **en dos casos:
  regresión y clasificación**.
- Registrar cada experimento y **guardar** el mejor modelo con **MLflow**.

> La **detección de anomalías / outliers** (Isolation Forest, DBSCAN) se trata
> ahora en el **Módulo 1** como tema de feature engineering / limpieza de datos,
> junto con los métodos estadísticos (z-score, IQR, MAD).

## Los cinco notebooks

| # | Notebook | Qué aprendes |
|---|---|---|
| 01 | `notebooks/01_regularization.ipynb` | Sobreajuste y sesgo–varianza; **Ridge (L2)** forma cerrada, geometría de la esparsidad de **Lasso (L1)**, **ElasticNet**; entrenado en California housing, registrado. |
| 02 | `notebooks/02_ensembles.ipynb` | **Bagging vs boosting** (matemática varianza/sesgo), **BaggingClassifier** (árbol solo vs embolsado + score OOB), **Random Forest**, luego **XGBoost / LightGBM / CatBoost** con una tabla detallada de *cómo difieren*; además **votación** (dura / suave / ponderada) y una sección de **stacking / blending** (aprendices base diversos + meta-aprendiz logístico vía OOF de 5 folds); benchmark sobre breast cancer. |
| 03 | `notebooks/03_svm.ipynb` | Margen máximo, **primal y dual**, pérdida hinge, margen blando `C`, **truco del kernel** (lineal/RBF/poly), fronteras de decisión sobre `make_moons`, ajustadas con GridSearch. |
| 04 | `notebooks/04_unsupervised.ipynb` | **DBSCAN vs OPTICS** (gráfico de alcanzabilidad), **KMeans vs GMM** (matemática EM), **t-SNE vs UMAP** sobre digits; métricas y gráficas como artefactos. |
| 05 | `notebooks/05_neural_networks_pytorch.ipynb` | **Notebook clave — muy ampliado.** De la **neurona simple** al **MLP** con **diagramas dibujados en código** (sin imágenes externas): activaciones y derivadas, **desvanecimiento del gradiente**, **decisiones de arquitectura antes de programar** (checklist), forward pass y **backprop**, el **algoritmo SGD explicado con imágenes** (superficie de pérdida, efecto de la learning rate, batch/mini-batch/estocástico), regularización (dropout/weight decay/early stopping/batchnorm), e **implementación en PyTorch en dos casos: regresión (lineal+MSE) y clasificación (sigmoid+BCE)**, registrados en MLflow. |

> Todos los textos de los notebooks están en **español**, con un balance entre
> intuición y matemática (el concepto primero, la fórmula compacta después).

## Arquitectura de MLflow

```
        ┌──────────────────────────────┐
        │   Notebooks Jupyter (01–05)  │
        │   utils/mlflow_helpers.py    │
        └───────────────┬──────────────┘
                        │  log_params / log_metrics / log_model
                        │  register_model
                        ▼
        ┌──────────────────────────────────────────────┐
        │        MLflow Tracking Server 3.x             │
        │        + Model Registry                       │
        │                                               │
        │   LOCAL:  docker compose (:5000)              │
        │   CLOUD:  DagsHub (https://dagshub.com/...)   │
        └──────────────────────────────────────────────┘
```

Los notebooks eligen el destino con un **backend** explícito
(`MLFLOW_BACKEND=local|dagshub`, configurable en un archivo `.env` — ver
`.env.example`), sin tocar código:

- **`local`** (default): servidor de docker compose en `MLFLOW_TRACKING_URI`
  (por defecto `http://localhost:5000`), con **fallback a un store SQLite local
  (`./mlruns.db`)** si no hay servidor disponible (SQLite sí soporta el Model
  Registry; el file store `./mlruns` quedó deprecado en MLflow ≥3.2).
- **`dagshub`**: MLflow gestionado de DagsHub, con URI y credenciales
  derivadas de `DAGSHUB_USER` / `DAGSHUB_REPO` / `DAGSHUB_TOKEN`.

Los helpers son
compatibles con **MLflow 2.x y 3.x**: usan la firma `name=` de `log_model` (la
posicional `artifact_path` quedó deprecada en MLflow 3) con fallback a la forma
antigua, y registran vía `log_model(registered_model_name=...)` (registrar
después con una URI `runs:/<id>/model` falla en MLflow 3 / DagsHub).

**Un run por modelo (default):** cada modelo genera exactamente un run curado
vía `log_and_register()`, que acepta `figures=`, `artifact_files=` e
`input_example=` (firma del modelo) y sube la configuración completa del
modelo como `model_config.json`. El notebook 05 sube como artefactos las
curvas de entrenamiento, el resumen del modelo y el **diagrama de la
arquitectura del MLP**, y envía las métricas por época en **un solo request**
(`log_batch`).

**Tracking exhaustivo (opt-in):** `setup_mlflow(..., autolog=True)` (o
`MLFLOW_AUTOLOG=1`) activa `mlflow.autolog()` — params/métricas/artefactos de
**cada** `fit()` de sklearn/xgboost/lightgbm (GridSearchCV sube su
`cv_results` completo). Ojo: crea un run extra por cada fit — ruidoso para
clase en vivo.

## MLflow 3.x

El stack de `mlflow/docker-compose.yml` construye el servidor sobre
`python:3.11-slim` instalando `mlflow>=3.1` + `psycopg2-binary` en arranque y lo
ejecuta con `mlflow server ... --serve-artifacts` sobre un backend Postgres con
volumen. `pyproject.toml` / `requirements.txt` fijan `mlflow>=3.1`.

> **Plataforma compartida.** El **Módulo 1** (su `airflow/` + `feast/`) levanta
> una plataforma compartida que **también** puede ofrecer MLflow en `:5000`. Usa
> **una u otra**, no ambas a la vez (las dos publican `:5000`). Ver
> `mlflow/README.md`.

## Prerrequisitos

- Python 3.10+ (3.11 recomendado)
- Docker + Docker Compose (para el servidor de MLflow; opcional — los notebooks
  corren sin él vía el fallback local)

## Instalación

Proyecto gestionado con **uv** (igual que el Módulo 1):

```bash
cd module2-advanced-ml
uv sync                  # crea .venv e instala todo (incl. xgboost/lightgbm/catboost/dagshub)
uv run jupyter lab       # ejecutar Jupyter dentro del entorno
```

Alternativa con pip clásico:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Arrancar MLflow

Elige **una** de las dos opciones; los notebooks funcionan igual con cualquiera
porque sólo dependen del backend configurado (`.env` / variables de entorno).

### Opción A — Local (Docker)

```bash
cd mlflow
docker compose up -d           # Postgres + servidor MLflow 3.x en :5000
# UI: http://localhost:5000
```

Es el backend por defecto (`MLFLOW_BACKEND=local`), no hay que configurar nada
más. Ver `mlflow/README.md` para detalles y resolución de problemas.

### Opción B — Cloud (DagsHub)

[DagsHub](https://dagshub.com) ofrece un servidor MLflow gestionado (con Model
Registry) por cada repositorio. Los helpers usan el cliente oficial — el
equivalente de:

```python
import dagshub
dagshub.init(repo_owner='abdala9512',
             repo_name='dsrp-machine-learning-engineering-20262',
             mlflow=True)

import mlflow
with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
```

`dagshub.init()` configura la URI de tracking y la autenticación (OAuth en el
navegador la primera vez; después usa el token cacheado). Para activarlo basta:

```bash
cd module2-advanced-ml
cp .env.example .env      # ya trae MLFLOW_BACKEND=dagshub y el repo del curso
```

`setup_mlflow()` carga el `.env` automáticamente y llama a `dagshub.init()` con
`DAGSHUB_USER` / `DAGSHUB_REPO` — no hace falta cambiar nada en los notebooks.
(`.env` está en `.gitignore`, así que nada sensible se sube al repo.)

Si no tienes el paquete `dagshub` instalado, los helpers caen a la vía manual:
define `DAGSHUB_TOKEN` en el `.env` (o `MLFLOW_TRACKING_URI` +
`MLFLOW_TRACKING_USERNAME` + `MLFLOW_TRACKING_PASSWORD`). También puedes forzar
el backend por notebook: `setup_mlflow("experimento", backend="dagshub")`.

## Ejecutar los notebooks

```bash
cd module2-advanced-ml
uv run jupyter lab   # abre notebooks/01_..05_ en orden
```

Con `backend=local`, si no hay ningún servidor MLflow accesible los notebooks
igual se ejecutan y registran en un SQLite local (`./mlruns.db`).

## Regenerar los notebooks

Los notebooks se generan (nunca se editan a mano como JSON) desde un único script:

```bash
python3 _build_notebooks.py
# validar
python3 -c "import nbformat, glob; [nbformat.read(f, as_version=4) for f in glob.glob('notebooks/*.ipynb')]; print('OK')"
```

## Archivos

```
module2-advanced-ml/
├── README.md                  ← este archivo
├── .env.example               ← plantilla de configuración (backend local/dagshub)
├── pyproject.toml              ← proyecto uv (uv sync / uv run jupyter lab)
├── uv.lock
├── requirements.txt            ← alternativa pip
├── _build_notebooks.py        ← generador nbformat (fuente de verdad)
├── mlflow/
│   ├── docker-compose.yml      ← Postgres + servidor MLflow 3.x (:5000) — opción local
│   └── README.md
├── notebooks/
│   ├── 01_regularization.ipynb
│   ├── 02_ensembles.ipynb
│   ├── 03_svm.ipynb
│   ├── 04_unsupervised.ipynb
│   └── 05_neural_networks_pytorch.ipynb
└── utils/
    └── mlflow_helpers.py       ← setup_mlflow / log_and_register / register_best_run
```
