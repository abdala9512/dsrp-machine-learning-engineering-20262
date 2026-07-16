"""Logica compartida de los DAGs hola-mundo (mismo patron que feature_pipeline.py).

Dos DAGs usan este modulo:

  * hello_feature_engineering      -> ciclo ONLINE  (materialize -> Redis -> serving)
  * hello_historical_features      -> ciclo OFFLINE (get_historical_features, point-in-time)

El dataset sintetico tiene HISTORIA: 5 usuarios x 7 dias de actividad ACUMULADA
(clicks y purchases solo crecen). Con historia, la diferencia entre los dos
mundos se ve clara:

  * offline store (parquet)  -> guarda TODOS los dias -> sets de entrenamiento
                                point-in-time ("que sabia de este usuario ESE dia")
  * online store (Redis)     -> guarda SOLO el ultimo valor -> serving en ms

El repo de Feast vive versionado en ``hello_feature_repo/`` (proyecto
`hello_features`); su registry y su parquet van al volumen compartido
``/feast_data``. El registry es COMPARTIDO con el proyecto de housing
(`module1_features`): la UI de Feast (http://localhost:8888) muestra ambos
proyectos en su selector.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# --- Rutas compartidas con los contenedores (ver docker-compose volumes) ------
FEAST_BIN = os.environ.get("FEAST_BIN", "feast")
# El SDK de feast vive en el mismo venv aislado que su CLI.
FEAST_PYTHON = str(Path(FEAST_BIN).parent / "python")

REPO_SRC = Path(os.environ.get("HELLO_FEAST_REPO_SRC", "/opt/airflow/hello_feature_repo_src"))
REPO_DIR = Path(os.environ.get("HELLO_FEAST_REPO_DIR", "/opt/airflow/hello_feast_repo"))
# Volumen compartido con feast-ui: aqui van el registry compartido y los parquet.
DATA_DIR = Path(os.environ.get("HELLO_FEAST_DATA", "/feast_data"))

RAW_PATH = DATA_DIR / "user_raw.parquet"
FEATURES_PATH = DATA_DIR / "user_features.parquet"
TRAINING_SET_PATH = DATA_DIR / "hello_training_set.parquet"

REDIS_HOST = os.environ.get("FEAST_REDIS_HOST", "redis")
REDIS_PORT = os.environ.get("FEAST_REDIS_PORT", "6379")

N_DAYS = 7


def create_dataset() -> str:
    """Genera el dataset crudo: 5 usuarios x 7 dias de actividad ACUMULADA.

    Deterministico (mismos valores en cada corrida); solo los timestamps se
    anclan a "ahora" para que la materializacion incremental y el TTL funcionen
    (y para que cada corrida escriba timestamps mas nuevos que la anterior:
    el online store de Feast ignora escrituras con event_timestamp mas viejo
    que el ya guardado). day 0 = hace 6 dias ... day 6 = ahora.
    """
    base_clicks = {1: 20, 2: 8, 3: 50, 4: 2, 5: 11}
    base_purchases = {1: 2, 2: 1, 3: 4, 4: 0, 5: 1}

    today = datetime.now(timezone.utc)
    rows = []
    for uid in base_clicks:
        clicks, purchases = 0, 0
        for day in range(N_DAYS):
            ts = today - timedelta(days=N_DAYS - 1 - day)
            clicks += base_clicks[uid] + (uid * day) % 7
            purchases += base_purchases[uid] + (1 if (uid + day) % 3 == 0 else 0)
            rows.append(
                {"user_id": uid, "event_timestamp": ts, "clicks": clicks, "purchases": purchases}
            )

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PATH, index=False)
    print(f"[create_dataset] {df['user_id'].nunique()} usuarios x {N_DAYS} dias = {len(df)} filas -> {RAW_PATH}")
    print(df[df["user_id"] == 1].to_string(index=False))  # muestra: la historia del usuario 1
    return str(RAW_PATH)


def build_features() -> str:
    """La 'ingenieria de features' del hola-mundo: UNA feature derivada."""
    df = pd.read_parquet(RAW_PATH)

    # Feature derivada: tasa de conversion (compras por click) hasta ese dia.
    df["conversion_rate"] = (df["purchases"] / df["clicks"]).astype("float32")
    df["clicks"] = df["clicks"].astype("int64")
    df["purchases"] = df["purchases"].astype("int64")

    # Columna de bookkeeping que Feast exige en la fuente (event_timestamp ya viene).
    df["created"] = pd.Timestamp(datetime.now(timezone.utc))

    df.to_parquet(FEATURES_PATH, index=False)
    print(f"[transform] {len(df)} filas -> {FEATURES_PATH}")
    return str(FEATURES_PATH)


def prepare_feast_repo() -> str:
    """Renderiza la copia del repo de Feast que usan los contenedores.

    Copia ``features.py`` del repo versionado y escribe un ``feature_store.yaml``
    con el host del servicio ``redis`` y el registry en el volumen compartido
    (asi la UI de Feast del puerto 8889 ve lo que el DAG aplica).
    """
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_SRC / "features.py", REPO_DIR / "features.py")

    yaml_text = f"""# Auto-generado por los DAGs hola-mundo para ejecucion en contenedor.
project: hello_features
provider: local
registry: {DATA_DIR / 'registry.db'}
online_store:
  type: redis
  connection_string: "{REDIS_HOST}:{REDIS_PORT}"
offline_store:
  type: file
entity_key_serialization_version: 3
"""
    (REPO_DIR / "feature_store.yaml").write_text(yaml_text)
    print(f"[prepare] repo en {REPO_DIR} (registry={DATA_DIR / 'registry.db'}, online={REDIS_HOST}:{REDIS_PORT})")
    return str(REPO_DIR)


def run_feast(*args: str) -> str:
    """Corre un comando del CLI de feast dentro del repo y muestra su salida."""
    cmd = [FEAST_BIN, "--chdir", str(REPO_DIR), *args]
    print(f"[feast] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"feast {' '.join(args)} fallo (exit {result.returncode})")
    return result.stdout


def run_feast_python(script_name: str, script_source: str) -> str:
    """Corre un script con el python del venv de feast (el SDK vive alli).

    OJO: el script se escribe FUERA del repo de Feast. `feast apply` importa
    todos los .py del repo buscando definiciones, asi que un script suelto
    adentro se ejecutaria en cada apply.
    """
    scripts_dir = REPO_DIR.parent / "hello_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / script_name
    script.write_text(script_source)
    cmd = [FEAST_PYTHON, str(script)]
    print(f"[feast-python] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_DIR)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"{script_name} fallo (exit {result.returncode})")
    return result.stdout
