"""Definiciones de Feast del hola-mundo: 1 entidad, 1 fuente, 1 feature view.

Version minima del patron de ``feature_repo/features.py`` (el repo real de
housing). El parquet fuente lo escriben los DAGs hola-mundo en el volumen
compartido ``HELLO_FEAST_DATA`` (dentro de Docker: /feast_data); si la
variable no esta definida (uso desde el host), cae a ./data junto a este archivo.
"""

import os
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

DATA_DIR = Path(os.environ.get("HELLO_FEAST_DATA", str(Path(__file__).resolve().parent / "data")))

# La "cosa" a la que se pegan las features: un usuario, identificado por user_id.
user = Entity(
    name="user",
    join_keys=["user_id"],
    value_type=ValueType.INT64,
    description="Un usuario del hola-mundo, identificado por user_id.",
)

# De donde salen los valores historicos: el parquet que escriben los DAGs.
# Contiene VARIOS dias por usuario -> el offline store guarda la historia
# completa; a Redis solo se materializa el ultimo valor de cada usuario.
user_source = FileSource(
    name="user_source",
    path=str(DATA_DIR / "user_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
    description="Actividad diaria acumulada por usuario (escrita por los DAGs hola-mundo).",
)

# Grupo nombrado y versionable de features, listo para servir online.
user_stats = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=14),
    schema=[
        Field(name="clicks", dtype=Int64),
        Field(name="purchases", dtype=Int64),
        Field(name="conversion_rate", dtype=Float32),
    ],
    online=True,
    source=user_source,
    tags={"team": "ml-platform", "module": "1", "demo": "hello-world"},
)
