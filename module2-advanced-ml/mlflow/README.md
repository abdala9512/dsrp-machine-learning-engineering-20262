# MLflow Tracking Stack (opción local)

This folder spins up a production-style **MLflow** deployment (MLflow **3.x**)
that the Module 2 notebooks can use as their tracking server.

> **¿Local o cloud?** Los notebooks eligen backend vía `MLFLOW_BACKEND`
> (`local` | `dagshub`), configurable en `module2-advanced-ml/.env` (ver
> `.env.example`). Este compose es la **opción local** (`MLFLOW_BACKEND=local`,
> el default); la alternativa **cloud** es **DagsHub**
> (`MLFLOW_BACKEND=dagshub` — los helpers llaman `dagshub.init(repo_owner=...,
> repo_name=..., mlflow=True)` con `DAGSHUB_USER` / `DAGSHUB_REPO`). Ver el
> README del módulo. Si usas DagsHub no necesitas levantar nada de esta
> carpeta.

> **Nota — plataforma compartida del curso.** El Módulo 1 (su carpeta
> `airflow/` + `feast/`) levanta una plataforma compartida que **también** puede
> ofrecer MLflow en el puerto `:5000`. Usa **una u otra**, no ambas a la vez:
> las dos publican `:5000` y entrarían en conflicto. Si ya tienes la plataforma
> del Módulo 1 corriendo, apunta los notebooks a ese MLflow vía
> `MLFLOW_TRACKING_URI` y no levantes este compose; si prefieres un MLflow
> dedicado al Módulo 2, usa este y deja la plataforma compartida apagada.

El servidor se construye sobre `python:3.11-slim` instalando `mlflow>=3.1` y
`psycopg2-binary` en arranque (en lugar de fijar una etiqueta de imagen
publicada), y se ejecuta con `mlflow server ... --serve-artifacts`.

## Architecture

```
   ┌────────────────────┐        ┌──────────────────────┐
   │  Jupyter notebooks │  HTTP  │   MLflow Tracking     │
   │  (the experiments) │ ─────► │   Server  :5000       │
   └────────────────────┘        │  + Model Registry     │
                                  └──────────┬───────────┘
                                             │
                       ┌─────────────────────┴──────────────────────┐
                       ▼                                             ▼
            ┌────────────────────┐                       ┌────────────────────┐
            │  Postgres (16)     │                       │  Artifact store     │
            │  backend store:    │                       │  docker volume      │
            │  params, metrics,  │                       │  /mlflow/artifacts  │
            │  runs, registry    │                       │  (models, plots)    │
            └────────────────────┘                       └────────────────────┘
```

- **Backend store (Postgres)** — relational DB holding run metadata, params,
  metrics, tags and **all Model Registry** information. The registry *requires*
  a DB-backed store (a plain file store cannot register models).
- **Artifact store (docker volume)** — large binary outputs: serialized models,
  plots, etc. Served through the tracking server (`--serve-artifacts`) so
  clients never need direct filesystem access.

## Start the stack

```bash
cd module2-advanced-ml/mlflow
docker compose up -d
```

First boot takes ~60–90s (the server installs `mlflow>=3.1` + `psycopg2-binary`
and waits for Postgres to become healthy). Check status:

```bash
docker compose ps
docker compose logs -f mlflow-server
```

## Open the UI

http://localhost:5000

You will see the **Experiments** tab (runs logged by the notebooks) and the
**Models** tab (the Model Registry).

## Point notebooks at it

The notebooks read the env var `MLFLOW_TRACKING_URI` (default
`http://localhost:5000`). To be explicit:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
jupyter lab
```

If the server is **not** running, `utils/mlflow_helpers.py` automatically falls
back to a local SQLite store (`./mlruns.db`) so the notebooks still execute
(SQLite even supports the Model Registry, unlike the deprecated `./mlruns`
file store).

## Stop / clean up

```bash
docker compose down       # stop containers, keep data
docker compose down -v    # stop AND delete volumes (Postgres + artifacts)
```

## Troubleshooting

- **Port 5000 in use (macOS AirPlay):** disable *AirPlay Receiver* in System
  Settings, or remap the published port to e.g. `5001:5000` and set
  `MLFLOW_TRACKING_URI=http://localhost:5001`.
- **Healthcheck failing:** give it `start_period` time; check
  `docker compose logs mlflow-server` for the `psycopg2` install step.
