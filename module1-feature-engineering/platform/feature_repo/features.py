"""Feast feature definitions for Module 1.

This file is discovered by `feast apply`. It declares:

  * an Entity            -> the join key features are attached to (passenger_id)
  * a FileSource         -> where the raw/engineered rows live (a parquet file)
  * a FeatureView        -> a named, reusable group of features with a TTL

The parquet file is produced by notebook 02 (or by running this module's
data-prep step). Each row must contain the entity column, an event timestamp
column, and one column per feature.

Compatible with Feast >= 0.40 (current API: Entity / FeatureView / FileSource /
Field, types imported from feast.types).
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, FileSource, Field, ValueType
from feast.types import Float32, Int64, String

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve the parquet path relative to this file so `feast apply` works no
# matter the current working directory.
DATA_DIR = Path(__file__).resolve().parent / "data"
PARQUET_PATH = str(DATA_DIR / "titanic_features.parquet")

# ---------------------------------------------------------------------------
# Entity: the "thing" we attach features to.
# ---------------------------------------------------------------------------
passenger = Entity(
    name="passenger",
    join_keys=["passenger_id"],
    value_type=ValueType.INT64,
    description="A Titanic passenger, identified by passenger_id.",
)

# ---------------------------------------------------------------------------
# Source: the offline data backing the features (a parquet file).
# ---------------------------------------------------------------------------
titanic_source = FileSource(
    name="titanic_source",
    path=PARQUET_PATH,
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
    description="Engineered Titanic features written by the Module 1 pipeline.",
)

# ---------------------------------------------------------------------------
# FeatureView: a reusable, named group of features tied to an entity + source.
# TTL controls how far back materialization/serving will look for a value.
# ---------------------------------------------------------------------------
titanic_passenger_features = FeatureView(
    name="titanic_passenger_features",
    entities=[passenger],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age", dtype=Float32),
        Field(name="fare", dtype=Float32),
        Field(name="fare_log", dtype=Float32),
        Field(name="age_scaled", dtype=Float32),
        Field(name="family_size", dtype=Int64),
        Field(name="pclass", dtype=Int64),
        Field(name="sex", dtype=String),
        Field(name="embark_town", dtype=String),
        Field(name="is_alone", dtype=Int64),
    ],
    online=True,
    source=titanic_source,
    tags={"team": "ml-platform", "module": "1"},
)
