# Plataforma compartida del curso (`platform/`)

El **Modulo 1** no solo ensena feature engineering (sobre el dataset **Ames
Housing**, prediciendo `SalePrice` — una tarea de **regresion**): tambien
**introduce la infraestructura compartida** que usa todo el curso. Esta carpeta
`platform/` reune en un unico `docker-compose.yml` las piezas que se repiten en
los modulos. En el Modulo 1 son dos:

1. **Feature store** (Feast) — definir, versionar y servir features.
2. **Orquestacion** (Apache Airflow) — el **orquestador compartido del curso**;
   cada modulo monta sus DAGs aqui.

> El **tracking de experimentos** con MLflow (registrar parametros, metricas y
> modelos) se introduce en el **Modulo 2**, cuando entramos de lleno al
> entrenamiento de modelos. El Modulo 1 = Feast + Airflow.

> Idea central: una sola plataforma, muchos modulos. Las features que defines
> aqui y los DAGs que programas en Airflow son los mismos componentes que veras
> en produccion.

```
platform/
├── docker-compose.yml        # TODA la infraestructura compartida
├── Dockerfile.airflow        # imagen de Airflow 3 (deps del Modulo 1 + feast en venv)
├── Dockerfile.feast          # imagen de la UI de Feast
├── serve_ui.py               # sirve la UI de Feast (agrega la ruta /api/v1/registry)
├── requirements-airflow.txt  # deps de los DAGs
├── README.md                 # este archivo
├── feature_repo/             # repo de Feast del pipeline real (housing)
├── hello_feature_repo/       # repo de Feast de los DAGs hola-mundo (hello_features)
└── dags/                     # DAGs del Modulo 1 (pipelines + sus modulos de logica)
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

   Airflow (orquestador)  -- entrena + evalua -->  modelo (joblib en ./data)
   programa el pipeline                            metricas en los logs de la tarea
```

### Servicios (configuracion ligera — 3 contenedores por defecto)

| Servicio | Imagen / build | Rol | Puerto (host) |
|---|---|---|---|
| `redis` | `redis:7` | **Online store** de Feast (serving) | 6379 |
| `feast-ui` | build `Dockerfile.feast` | **UI web de Feast** — proyectos housing y hola-mundo | 8888 |
| `airflow` | build `Dockerfile.airflow` | **Airflow 3 en un solo contenedor** (`airflow standalone`, SQLite) | 8080 |
| `redisinsight` *(opcional)* | `redis/redisinsight` | **UI web de Redis** — solo con `--profile redis-ui` | 5540 |
| `postgres` *(opcional)* | `postgres:16` | Registry / offline SQL de Feast — solo con `--profile feast-sql-registry` | 5432 |

**URIs importantes**
- Online store de Feast dentro de la red: host `redis` (no `localhost`).
- **UI de Feast: `http://localhost:8888`** — explora el registry (entidades,
  feature views, fuentes y features). El registry es **compartido** y contiene
  DOS proyectos: `module1_features` (housing) y `hello_features` (los DAGs
  hola-mundo); cambia entre ellos con el **selector de proyectos** de la UI.
  El contenedor corre `feast apply` sobre ambos repos al arrancar, asi que la
  UI muestra las definiciones aunque todavia no hayas materializado datos.
  > Nota tecnica: la UI de feast 0.64 pide el registry en `/api/v1/registry`,
  > ruta que su servidor ya no expone (la UI carga en blanco). `serve_ui.py`
  > arranca la app oficial y le agrega esa ruta de vuelta.
- **Airflow UI: `http://localhost:8080`** — login **`admin` / `airflow`**.
- **UI de Redis (opcional): `http://localhost:5540`** (RedisInsight, con
  `--profile redis-ui`). Redis no trae UI propia; conecta al host `redis`, puerto
  6379, para inspeccionar las claves del online store.

> **Configuracion ligera (a proposito).** Para no fundir tu maquina:
> - **Airflow 3 en UN solo contenedor** via `airflow standalone` (api-server +
>   scheduler + dag-processor + triggerer) sobre **SQLite** (antes eran 4
>   contenedores). Es un setup de *desarrollo*: perfecto para uno o dos DAGs
>   pequenos.
> - La **imagen de Airflow instala solo las deps del Modulo 1** (pandas,
>   scikit-learn, joblib). Adios a `torch`/`sentence-transformers`/`xgboost`/
>   `mlflow`: la imagen paso de varios GB a unos cientos de MB.
> - **`feast` vive en un venv aislado** dentro de la imagen (`/opt/feast-venv`):
>   feast fija `uvicorn<=0.34` y Airflow 3 exige `uvicorn>=0.37`, asi que no pueden
>   convivir en el mismo entorno. El DAG llama a feast por su binario (`FEAST_BIN`).
> - El **Postgres de Feast es opcional** (el registry por defecto es un archivo).

> **Experiment tracking en el Modulo 2.** En el Modulo 1 no hay servidor MLflow.
> El DAG entrena, evalua y guarda el modelo en disco; el tracking de experimentos
> con MLflow se introduce en el Modulo 2.

---

## (Opcional) Orquestar tambien otros modulos

Para mantener la plataforma ligera, por defecto Airflow **solo monta los DAGs del
Modulo 1**. Si quieres el orquestador compartido de todo el curso, en
`docker-compose.yml` descomenta los montajes de DAGs de otros modulos y, en
`requirements-airflow.txt`, sus dependencias (la imagen crecera bastante — `torch`
pesa varios GB):

```yaml
# docker-compose.yml (servicio airflow -> volumes)
- ../../module3-time-series/airflow/dags:/opt/airflow/dags/module3:ro
- ../../module4-genai/airflow/dags:/opt/airflow/dags/module4:ro
```

---

## Levantar la plataforma

```bash
cd module1-feature-engineering/platform

# Todo lo ligero (redis + feast-ui + airflow):
docker compose up -d --build

# Solo la capa Feast + su UI (flujo manual / notebooks, sin Airflow):
docker compose up -d --build redis feast-ui            # UI en http://localhost:8888

# Con la UI de Redis (RedisInsight) ademas:
docker compose --profile redis-ui up -d --build       # UI en http://localhost:5540

# Con el Postgres de Feast (registry SQL) ademas:
docker compose --profile feast-sql-registry up -d --build

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

## Hola mundo: Airflow + Feast (empieza por aqui)

Antes del pipeline real de housing hay DOS DAGs hola-mundo, uno por cada mitad
de un feature store. Comparten el dataset (5 usuarios × 7 dias de actividad
**acumulada**), el repo de Feast versionado en `hello_feature_repo/` (proyecto
`hello_features`) y la logica en `dags/hello_pipeline.py`:

**1. `hello_feature_engineering` — el ciclo ONLINE (serving):**

```
 create_dataset -> transform -+
                              +-> feast_apply -> feast_materialize -> read_online_features
 prepare_feast_repo ---------+
```

| Tarea | Que hace |
|------|----------|
| `create_dataset` | 5 usuarios × 7 dias de `clicks`/`purchases` acumulados → parquet |
| `transform` | Ingenieria de features: `conversion_rate` + timestamps de Feast |
| `prepare_feast_repo` | Renderiza `hello_feature_repo/` para Docker (host `redis`, registry compartido) |
| `feast_apply` | Registra entidad `user` y feature view `user_stats` |
| `feast_materialize` | Carga el ULTIMO valor por usuario a Redis (online store) |
| `read_online_features` | `get_online_features()` desde Redis — como un servicio de inferencia |

**2. `hello_historical_features` — el ciclo OFFLINE (entrenamiento):**
mismos primeros 4 pasos, pero termina en `get_historical_features`: un *entity
dataframe* con pares (`user_id`, `event_timestamp`) produce un **set de
entrenamiento point-in-time**. Como el dataset es acumulado, el efecto se ve a
simple vista en los logs: el usuario 1 pedido en 3 fechas devuelve 3 valores
distintos de `clicks` (41 → 110 → 161), nunca uno "del futuro". Eso es lo que
evita el data leakage temporal.

```bash
docker compose up -d --build
# Abre http://localhost:8080 (admin/airflow), des-pausa los DAGs `hello_*`
# y dales ▶ Trigger (primero el online, luego el historico — o al reves; son
# independientes).
```

Donde ver los resultados:

- **Logs de las tareas** `read_online_features` / `get_historical_features` en
  la UI de Airflow.
- **UI de Feast: http://localhost:8888**, proyecto `hello_features` en el
  selector de proyectos (entidad `user`, feature view `user_stats`). El
  registry compartido vive en el volumen `feast-data`.
- **RedisInsight** (`--profile redis-ui`, http://localhost:5540): las claves del
  proyecto `hello_features` (busca con el patron `*hello_features`; los nombres
  de clave son binarios, es normal que se vean "raros").

Cuando las dos mitades queden claras, pasa al pipeline real:

---

## El pipeline cerrado (Modulo 1)

El DAG `feature_engineering_pipeline` (en `dags/`) convierte los pasos manuales
del notebook 03 en un pipeline programado y **cierra el ciclo hasta el modelo**:

```
 extract -+
          +-> transform -> validate -> feast_apply -> feast_materialize -> train_model
 prepare -+
```

| Tarea | Que hace |
|------|----------|
| `extract` | Lee el CSV de Ames Housing (`HOUSING_CSV`; fallback sintetico) → `data/housing_raw.parquet` |
| `prepare_feast_repo` | Renderiza un repo de Feast apuntando al servicio `redis` |
| `transform` | Ingenieria de features → `feast_repo/data/housing_features.parquet` |
| `validate` | Control de calidad: esquema, conteo, clave unica (`house_id`), sin nulos |
| `feast_apply` | `feast apply` — registra entidades / feature views |
| `feast_materialize` | `feast materialize-incremental <now>` — carga a Redis |
| `train_model` | Entrena + evalua un regresor (`SalePrice`) y lo **guarda en disco** |

`train_model` lee el parquet del offline store, entrena un `RandomForestRegressor`
que predice `SalePrice` (sobre `log1p`), evalua (RMSE / MAE / R²), deja las metricas
en los logs de la tarea y persiste el modelo (`data/housing_model.joblib`).

> El **experiment tracking** y el **registro de modelos** con MLflow se introducen
> en el **Modulo 2**.

### Ejecutarlo

```bash
docker compose up -d --build
# Abre http://localhost:8080 (airflow/airflow), des-pausa el DAG y dale ▶ Trigger.

# o por CLI:
docker compose exec airflow-scheduler airflow dags trigger feature_engineering_pipeline
```

Revisa los **logs de la tarea `train_model`** en la UI de Airflow para ver las
metricas (RMSE / MAE / R²).

---

## Notas y troubleshooting

- **`feature_store.yaml`** usa `registry: data/registry.db` (archivo) por
  defecto. Cambia al bloque `sql` comentado para usar el contenedor Postgres.
- **`feast materialize` "connection refused"** → el servicio `redis` debe estar
  `healthy`; el DAG usa `FEAST_REDIS_HOST=redis`.
- **Permisos en logs (Linux)** → fija tu UID antes de levantar:
  `echo "AIRFLOW_UID=$(id -u)" >> .env` en `platform/`, luego `docker compose up -d`.
- **Cambios de imagen no se reflejan** → `docker compose build --no-cache`.
- **Puerto 8080 ocupado** → cambia el puerto publicado en `docker-compose.yml`.

## Teardown

```bash
docker compose down          # detiene, conserva volumenes
docker compose down -v       # detiene y borra TODOS los datos
cd feature_repo && feast teardown   # limpia el registry / online store de Feast
```
