# Plataforma compartida del curso (`platform/`)

El **Modulo 1** no solo ensena feature engineering: tambien **introduce la
infraestructura compartida** que usa todo el curso. Esta carpeta `platform/`
reune en un unico `docker-compose.yml` las tres piezas que se repiten en todos
los modulos:

1. **Feature store** (Feast) — definir, versionar y servir features.
2. **Tracking de experimentos** (MLflow 3.x) — registrar parametros, metricas y
   modelos.
3. **Orquestacion** (Apache Airflow) — el **orquestador compartido del curso**;
   cada modulo monta sus DAGs aqui.

> Idea central: una sola plataforma, muchos modulos. Las features que defines
> aqui, los experimentos que registras en MLflow y los DAGs que programas en
> Airflow son los mismos componentes que veras en produccion.

```
platform/
├── docker-compose.yml        # TODA la infraestructura compartida
├── Dockerfile.airflow        # apache/airflow:2.10.5-python3.11 + deps de los 4 modulos
├── Dockerfile.mlflow         # python:3.11-slim + mlflow>=3.1 (servidor de tracking)
├── requirements-airflow.txt  # deps combinadas de los DAGs de TODOS los modulos
├── README.md                 # este archivo
├── feature_repo/             # repo de Feast (feature_store.yaml, features.py, data/)
└── dags/                     # DAGs del Modulo 1 (feature_pipeline.py + el DAG)
```

---

## Arquitectura

```
                        +----------------------------+
                        |        Feast SDK           |
                        | apply / materialize / get  |
                        +-------------+--------------+
                                      |
       escribe features               |   lee features
   (entrenamiento, point-in-time)     |   (serving, baja latencia)
                                      |
        +-----------------------------+----------------------------+
        |                                                          |
        v                                                          v
+-----------------------+        materialize        +---------------------------+
|  OFFLINE STORE        | ------------------------> |  ONLINE STORE             |
|  parquet (./data)     |   solo el ultimo valor    |  Redis  (host "redis")    |
|  historia completa    |                           |  un registro por entidad  |
+-----------------------+                           +---------------------------+

   Airflow (orquestador)  -- entrena -->  MLflow 3.x (tracking)  http://mlflow:5000
   programa el pipeline                   parametros / metricas / modelos
```

### Servicios

| Servicio | Imagen / build | Rol | Puerto (host) |
|---|---|---|---|
| `redis` | `redis:7` | **Online store** de Feast (serving) | 6379 |
| `postgres` | `postgres:16` | Registry / offline backend de Feast | 5432 |
| `mlflow` | build `Dockerfile.mlflow` | **Servidor MLflow 3.x** (tracking) | 5000 |
| `mlflow-db` | `postgres:16` | Backend de metadatos de MLflow | (interno) |
| `airflow-db` | `postgres:16` | Base de metadatos de Airflow | (interno) |
| `airflow-init` | build `Dockerfile.airflow` | One-shot: migracion + usuario admin | — |
| `airflow-webserver` | build `Dockerfile.airflow` | **UI de Airflow** | 8080 |
| `airflow-scheduler` | build `Dockerfile.airflow` | Ejecuta los DAGs | — |

**URIs importantes**
- MLflow desde el host: `http://localhost:5000`.
  Dentro de la red de Docker: `http://mlflow:5000` (lo usan los DAGs).
- Online store de Feast dentro de la red: host `redis` (no `localhost`).
- Airflow UI: `http://localhost:8080` (usuario `airflow` / `airflow`).

> **Imagen de Airflow PESADA.** `Dockerfile.airflow` instala las dependencias de
> los cuatro modulos (Feast, scikit-learn, XGBoost, statsmodels, MLflow,
> qdrant-client, sentence-transformers...). La primera construccion puede tardar
> varios minutos y pesar varios GB. Es a proposito: un solo orquestador para
> todo el curso.

---

## Como los modulos comparten Airflow

El `docker-compose.yml` monta los DAGs de cada modulo en un subfolder distinto
de `/opt/airflow/dags`:

```yaml
volumes:
  - ./dags:/opt/airflow/dags/module1
  - ../../module2-advanced-ml/airflow/dags:/opt/airflow/dags/module2
  - ../../module3-time-series/airflow/dags:/opt/airflow/dags/module3
  - ../../module4-genai/airflow/dags:/opt/airflow/dags/module4
```

Asi, al levantar la plataforma desde el Modulo 1, la UI de Airflow muestra los
DAGs de los cuatro modulos a la vez. Los servicios de Airflow exponen
`MLFLOW_TRACKING_URI=http://mlflow:5000` y `QDRANT_URL=http://host.docker.internal:6333`
(con `extra_hosts: host.docker.internal:host-gateway`) para que cualquier DAG
pueda registrar en MLflow o hablar con Qdrant.

---

## Levantar la plataforma

```bash
cd module1-feature-engineering/platform

# Todo (construye las imagenes de Airflow y MLflow):
docker compose up -d --build

# Solo la capa Feast (flujo manual / notebooks):
docker compose up -d redis postgres

# Solo MLflow:
docker compose up -d --build mlflow

docker compose ps          # espera a que los servicios esten "healthy"
```

Validar la configuracion sin levantar nada:

```bash
docker compose config -q   # debe pasar sin errores
```

---

## Instalar Feast (lado del host, para notebooks)

```bash
# El extra [redis] es OBLIGATORIO: trae el driver del online store de Redis.
pip install "feast[redis]" redis
```

> Un `pip install feast` "pelado" falla en `feast apply` con
> `Could not import module 'feast.infra.online_stores.redis'`.

---

## El pipeline cerrado (Modulo 1)

El DAG `feature_engineering_pipeline` (en `dags/`) convierte los pasos manuales
del notebook 02 en un pipeline programado y **cierra el ciclo hasta el modelo**:

```
 extract -+
          +-> transform -> validate -> feast_apply -> feast_materialize -> train_model
 prepare -+
```

| Tarea | Que hace |
|------|----------|
| `extract` | Carga Titanic crudo (fallback sintetico si no hay red) → `data/titanic_raw.parquet` |
| `prepare_feast_repo` | Renderiza un repo de Feast apuntando al servicio `redis` |
| `transform` | Ingenieria de features → `feast_repo/data/titanic_features.parquet` |
| `validate` | Control de calidad: esquema, conteo, clave unica, sin nulos |
| `feast_apply` | `feast apply` — registra entidades / feature views |
| `feast_materialize` | `feast materialize-incremental <now>` — carga a Redis |
| `train_model` | Entrena un clasificador simple y lo **registra en MLflow** |

`train_model` lee el parquet del offline store, entrena una `LogisticRegression`
que predice `survived`, evalua (accuracy / ROC-AUC) y registra parametros,
metricas y el modelo en `http://mlflow:5000` (experimento
`module1_feature_pipeline`).

### Ejecutarlo

```bash
docker compose up -d --build
# Abre http://localhost:8080 (airflow/airflow), des-pausa el DAG y dale ▶ Trigger.

# o por CLI:
docker compose exec airflow-scheduler airflow dags trigger feature_engineering_pipeline
```

Luego abre **http://localhost:5000** para ver el run, las metricas y el modelo
registrado.

---

## Notas y troubleshooting

- **`feature_store.yaml`** usa `registry: data/registry.db` (archivo) por
  defecto. Cambia al bloque `sql` comentado para usar el contenedor Postgres.
- **`feast materialize` "connection refused"** → el servicio `redis` debe estar
  `healthy`; el DAG usa `FEAST_REDIS_HOST=redis`.
- **Permisos en logs (Linux)** → fija tu UID antes de levantar:
  `echo "AIRFLOW_UID=$(id -u)" >> .env` en `platform/`, luego `docker compose up -d`.
- **Cambios de imagen no se reflejan** → `docker compose build --no-cache`.
- **Puerto 8080 / 5000 ocupado** → cambia el puerto publicado en `docker-compose.yml`.

## Teardown

```bash
docker compose down          # detiene, conserva volumenes
docker compose down -v       # detiene y borra TODOS los datos
cd feature_repo && feast teardown   # limpia el registry / online store de Feast
```
