"""
mlflow_helpers.py
=================

Small, reusable helpers shared by every notebook in Module 2.

Goals
-----
1. Point MLflow at one of two explicit **backends**:

   - ``local``   — the docker-compose stack (``http://localhost:5000``, or
     whatever ``MLFLOW_TRACKING_URI`` says). If the server is unreachable we
     transparently fall back to a **local SQLite store** (``./mlruns.db``) so
     the notebooks always run, even without Docker.
   - ``dagshub`` — the managed MLflow server that DagsHub attaches to every
     repo (``https://dagshub.com/<user>/<repo>.mlflow``). Uses the official
     ``dagshub`` client, equivalent to:

     .. code-block:: python

        import dagshub
        dagshub.init(repo_owner="abdala9512",
                     repo_name="dsrp-machine-learning-engineering-20262",
                     mlflow=True)

     ``dagshub.init`` configures the tracking URI and handles authentication
     (cached token, or an OAuth prompt in the browser the first time). If the
     ``dagshub`` package is not installed, we fall back to building the URI
     manually from ``DAGSHUB_TOKEN``.

   The backend is chosen by the ``backend=`` argument of :func:`setup_mlflow`,
   or the ``MLFLOW_BACKEND`` environment variable (default ``local``). All the
   configuration can live in a ``.env`` file next to this module's folder
   (see ``.env.example``) — no need to export anything in the shell:

   .. code-block:: ini

      MLFLOW_BACKEND=dagshub
      DAGSHUB_USER=abdala9512
      DAGSHUB_REPO=dsrp-machine-learning-engineering-20262
      # DAGSHUB_TOKEN only needed without the `dagshub` package (no OAuth)

2. Provide a single ``log_and_register`` convenience function that logs
   params + metrics + a fitted model and (optionally) registers it in the
   MLflow **Model Registry**.

3. Provide a ``register_best_run`` helper to promote the best run of an
   experiment into the registry.

These helpers keep the notebooks focused on *ML theory & code* instead of
boilerplate MLflow plumbing.
"""

from __future__ import annotations

import os
import socket
from contextlib import closing
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import mlflow


# --------------------------------------------------------------------------- #
# Tracking URI resolution
# --------------------------------------------------------------------------- #
def _load_env_file() -> None:
    """
    Load a ``.env`` file (simple ``KEY=VALUE`` lines) into ``os.environ`` so
    students can keep their DagsHub credentials out of the notebooks.

    Looks in: the module folder (``module2-advanced-ml/.env``), the current
    working directory, and its parent. Existing environment variables are
    NOT overridden. No external dependency (python-dotenv) required.
    """
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(module_dir, ".env"),
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.getcwd(), "..", ".env"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
        return  # first .env found wins


# Repo del curso — se puede sobreescribir con DAGSHUB_USER / DAGSHUB_REPO.
_DAGSHUB_DEFAULT_OWNER = "abdala9512"
_DAGSHUB_DEFAULT_REPO = "dsrp-machine-learning-engineering-20262"


def _setup_dagshub(tracking_uri: Optional[str]) -> str:
    """
    Connect MLflow to DagsHub and return the tracking URI.

    Preferred path — the official client:

        import dagshub
        dagshub.init(repo_owner=<owner>, repo_name=<repo>, mlflow=True)

    which sets the tracking URI and handles auth (cached token or an OAuth
    prompt on first use). Owner/repo come from ``DAGSHUB_USER`` /
    ``DAGSHUB_REPO`` (defaults: the course repo).

    Fallback path (``dagshub`` package not installed, or an explicit
    ``tracking_uri`` was given): build the URI manually and export
    ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD`` from
    ``DAGSHUB_TOKEN``.
    """
    owner = os.getenv("DAGSHUB_USER", _DAGSHUB_DEFAULT_OWNER)
    repo = os.getenv("DAGSHUB_REPO", _DAGSHUB_DEFAULT_REPO)

    if not tracking_uri:
        try:
            import dagshub  # type: ignore
        except ImportError:
            pass
        else:
            dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
            return mlflow.get_tracking_uri()

    # Manual fallback: explicit URI, or one built from env vars, plus a token.
    env_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        uri = tracking_uri
    elif "dagshub.com" in env_uri:
        uri = env_uri
    else:
        uri = f"https://dagshub.com/{owner}/{repo}.mlflow"

    if not os.getenv("MLFLOW_TRACKING_USERNAME"):
        token = os.getenv("DAGSHUB_TOKEN", "")
        if not token:
            raise ValueError(
                "backend='dagshub' needs credentials: install the `dagshub` "
                "package (pip install dagshub) for the OAuth flow, or set "
                "DAGSHUB_TOKEN (dagshub.com > Settings > Tokens) e.g. in "
                "module2-advanced-ml/.env."
            )
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    if not _server_is_up(uri):
        raise ConnectionError(
            f"[mlflow_helpers] DagsHub server '{uri}' is not reachable — "
            "check your internet connection and the repo URL. "
            "(Use backend='local' to work offline.)"
        )
    return uri


def _server_is_up(uri: str, timeout: float = 1.5) -> bool:
    """Return True if we can open a TCP connection to the http(s) tracking URI."""
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        # file:// or a bare path is always "up" (local store)
        return True
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with closing(socket.create_connection((host, port), timeout=timeout)):
            return True
    except OSError:
        return False


def setup_mlflow(
    experiment_name: str,
    backend: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    local_fallback: str = "./mlruns.db",
    autolog: bool = False,
) -> str:
    """
    Configure MLflow for a notebook and return the *effective* tracking URI.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment to create / use.
    backend : {"local", "dagshub"}, optional
        Where to track experiments. Defaults to the ``MLFLOW_BACKEND`` env var,
        or ``"local"``.

        - ``"local"``   — docker-compose server at ``MLFLOW_TRACKING_URI``
          (default ``http://localhost:5000``); falls back to a local SQLite
          store (``./mlruns.db``) if unreachable.
        - ``"dagshub"`` — DagsHub's managed MLflow, via
          ``dagshub.init(repo_owner=..., repo_name=..., mlflow=True)``.
          Owner/repo from ``DAGSHUB_USER`` / ``DAGSHUB_REPO`` (env or ``.env``
          file; defaults to the course repo). Auth: OAuth prompt / cached
          token, or ``DAGSHUB_TOKEN`` if the ``dagshub`` package is missing.
    tracking_uri : str, optional
        Explicit tracking URI; overrides the backend's default resolution.
    local_fallback : str
        SQLite file used as local store when the local server is unreachable.
        (SQLite rather than a ``./mlruns`` file store: MLflow ≥3.2 deprecates
        the filesystem backend, and SQLite also supports the Model Registry.)
    autolog : bool
        Enable ``mlflow.autolog(log_models=False)`` (default **False**; also
        via env ``MLFLOW_AUTOLOG=1``). OJO: crea un run extra por CADA
        ``fit()`` de sklearn (incluidos scalers y cada candidato de un
        GridSearchCV) — exhaustivo pero ruidoso. Apagado, cada modelo genera
        exactamente **un** run curado vía ``log_and_register``.

    Returns
    -------
    str
        The tracking URI that was actually set.
    """
    _load_env_file()

    backend = (backend or os.getenv("MLFLOW_BACKEND", "local")).strip().lower()
    if backend not in ("local", "dagshub"):
        raise ValueError(
            f"Unknown backend '{backend}' — expected 'local' or 'dagshub'."
        )

    if backend == "dagshub":
        uri = _setup_dagshub(tracking_uri)
    else:
        uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        if not _server_is_up(uri):
            fallback = os.path.abspath(local_fallback)
            print(
                f"[mlflow_helpers] Tracking server '{uri}' not reachable — "
                f"falling back to local SQLite store at '{fallback}'."
            )
            uri = f"sqlite:///{fallback}"

    mlflow.set_tracking_uri(uri)

    # Un experimento BORRADO (papelera) mantiene reservado su nombre y hace
    # fallar set_experiment. Si es el caso, lo restauramos automáticamente.
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        mlflow.MlflowClient().restore_experiment(exp.experiment_id)
        print(
            f"[mlflow_helpers] El experimento '{experiment_name}' estaba borrado "
            "(su nombre queda reservado en MLflow) — lo restauré para poder "
            "usarlo. Para empezar 100% de cero usa otro nombre de experimento."
        )
    mlflow.set_experiment(experiment_name)

    if autolog or os.getenv("MLFLOW_AUTOLOG", "0").lower() in ("1", "true"):
        try:
            mlflow.autolog(log_models=False, silent=True)
            print("[mlflow_helpers] Autolog      : ON (un run extra por cada fit)")
        except Exception as exc:  # nunca rompemos el notebook por el autolog
            print(f"[mlflow_helpers] Autolog no disponible: {exc}")

    print(f"[mlflow_helpers] Backend      : {backend}")
    print(f"[mlflow_helpers] Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"[mlflow_helpers] Experiment   : {experiment_name}")
    return uri


def registry_available() -> bool:
    """
    The Model Registry requires a database-backed store. The local ``file://``
    store does NOT support it, so notebooks should skip registration there.
    """
    uri = mlflow.get_tracking_uri()
    return urlparse(uri).scheme in ("http", "https") or uri.startswith(
        ("postgresql", "mysql", "sqlite", "mssql")
    )


# --------------------------------------------------------------------------- #
# Logging + registration
# --------------------------------------------------------------------------- #
def _log_model_compat(
    log_fn: Any,
    model: Any,
    artifact_path: str,
    registered_model_name: Optional[str] = None,
    input_example: Any = None,
):
    """Call an MLflow ``log_model`` flavor function across MLflow 2.x and 3.x.

    In MLflow 3 the positional ``artifact_path`` argument is deprecated in favor
    of the keyword ``name=``. We try the modern ``name=`` signature first and
    fall back to the positional form on older MLflow, so the helper works on
    both 2.x and 3.x.

    Registration is delegated to ``log_model(registered_model_name=...)``
    rather than a separate ``mlflow.register_model("runs:/...")`` call: in
    MLflow 3 models live as *logged models* instead of plain run artifacts,
    and resolving a hand-built ``runs:/<id>/<path>`` URI fails against servers
    (e.g. DagsHub) whose logged-model search is unsupported — the dreaded
    "Unable to find a logged_model with artifact_path ... under run ...".
    """
    kwargs = {}
    if registered_model_name:
        kwargs["registered_model_name"] = registered_model_name
    if input_example is not None:
        kwargs["input_example"] = input_example
    try:
        return log_fn(model, name=artifact_path, **kwargs)
    except TypeError:
        # Older MLflow (2.x) does not accept name=; use the positional form.
        return log_fn(model, artifact_path, **kwargs)


def _log_model(
    model: Any,
    artifact_path: str,
    flavor: str,
    registered_model_name: Optional[str] = None,
    input_example: Any = None,
):
    """Log a model using the appropriate MLflow flavor (MLflow 2.x/3.x safe)."""
    flavors = {
        "sklearn": "mlflow.sklearn",
        "xgboost": "mlflow.xgboost",
        "lightgbm": "mlflow.lightgbm",
        "pytorch": "mlflow.pytorch",
        "catboost": "mlflow.catboost",
    }
    import importlib

    module = importlib.import_module(flavors.get(flavor, "mlflow.sklearn"))
    return _log_model_compat(
        module.log_model, model, artifact_path, registered_model_name, input_example
    )


def log_and_register(
    *,
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model: Any = None,
    artifact_path: str = "model",
    flavor: str = "sklearn",
    registered_model_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    input_example: Any = None,
    figures: Optional[Dict[str, Any]] = None,
    artifact_files: Optional[Dict[str, str]] = None,
) -> str:
    """
    Start an MLflow run, log params/metrics/model and optionally register it.

    Besides the curated ``params``/``metrics``, the run is enriched with:

    - ``figures``: dict ``{"plots/nombre.png": matplotlib_figure}`` logged as
      image artifacts via ``mlflow.log_figure``.
    - ``artifact_files``: dict ``{path_local: carpeta_artefacto_o_""}`` logged
      via ``mlflow.log_artifact``.
    - ``input_example``: a small input sample; MLflow infers and stores the
      model **signature** with it.
    - the model's full configuration (``get_params()`` for sklearn-likes,
      ``repr`` for PyTorch modules) as ``model_config.json`` / ``.txt``.

    Returns the ``run_id``.
    """
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if figures:
            for fname, fig in figures.items():
                mlflow.log_figure(fig, fname)
        if artifact_files:
            for path, dest in artifact_files.items():
                mlflow.log_artifact(path, dest or None)

        # Configuración completa del modelo como artefacto ("todo lo posible")
        if model is not None:
            try:
                if hasattr(model, "get_params"):
                    mlflow.log_dict(
                        {k: str(v) for k, v in model.get_params(deep=True).items()},
                        "model_config.json",
                    )
                else:
                    mlflow.log_text(str(model), "model_config.txt")
            except Exception:
                pass

        if model is not None:
            register_as = (
                registered_model_name
                if registered_model_name and registry_available()
                else None
            )
            info = _log_model(model, artifact_path, flavor, register_as, input_example)
            # Etiquetamos el run con el model_id del *logged model* (MLflow 3):
            # es la única forma fiable de registrarlo DESPUÉS en servidores cuyo
            # search_logged_models no devuelve resultados (p. ej. DagsHub) —
            # register_best_run() usa esta etiqueta.
            model_id = getattr(info, "model_id", None)
            if model_id:
                mlflow.set_tag("model_id", str(model_id))

            if register_as:
                print(
                    f"[mlflow_helpers] Registered '{register_as}' "
                    f"from run {run.info.run_id}"
                )
            elif registered_model_name:
                print(
                    "[mlflow_helpers] Registry unavailable on the local file "
                    "store — skipping model registration. Start the MLflow "
                    "server (docker compose up) to enable it."
                )
        return run.info.run_id


def register_best_run(
    experiment_name: str,
    metric: str,
    registered_model_name: str,
    mode: str = "max",
    artifact_path: str = "model",
) -> Optional[str]:
    """
    Find the best run of an experiment by ``metric`` and register its model.

    Parameters
    ----------
    metric : str
        Metric key to rank runs by.
    mode : {"max", "min"}
        Whether higher or lower is better.

    Returns
    -------
    str | None
        The version string of the newly registered model, or None if the
        registry is unavailable / no runs found.
    """
    if not registry_available():
        print(
            "[mlflow_helpers] Registry unavailable on local file store — "
            "skipping register_best_run()."
        )
        return None

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"[mlflow_helpers] Experiment '{experiment_name}' not found.")
        return None

    order = "DESC" if mode == "max" else "ASC"
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if runs.empty:
        print("[mlflow_helpers] No runs found to register.")
        return None

    best_run_id = runs.iloc[0]["run_id"]
    best_value = runs.iloc[0][f"metrics.{metric}"]

    # Vía preferida: la etiqueta `model_id` que log_and_register deja en el run
    # apunta directo al logged model (models:/<id>) — funciona en cualquier
    # servidor, incluido DagsHub, sin depender de resolver URIs runs:/ ni de
    # search_logged_models (que en DagsHub devuelve vacío).
    tag_model_id = str(runs.iloc[0].get("tags.model_id") or "")
    if tag_model_id and tag_model_id != "nan":
        mv = mlflow.register_model(f"models:/{tag_model_id}", registered_model_name)
    else:
        try:
            # MLflow 2.x (y servidores que aún resuelven artefactos de run)
            mv = mlflow.register_model(
                f"runs:/{best_run_id}/{artifact_path}", registered_model_name
            )
        except Exception:
            # MLflow 3.x: el modelo vive como *logged model* — lo buscamos por
            # source_run_id y registramos models:/<id>.
            models = mlflow.search_logged_models(
                experiment_ids=[exp.experiment_id],
                filter_string=f"source_run_id = '{best_run_id}'",
                output_format="list",
            )
            if not models:
                raise
            mv = mlflow.register_model(
                f"models:/{models[0].model_id}", registered_model_name
            )
    print(
        f"[mlflow_helpers] Best run {best_run_id} ({metric}={best_value:.4f}) "
        f"registered as '{registered_model_name}' v{mv.version}"
    )
    return mv.version
