"""Sirve la UI de Feast con el endpoint de registry que le falta a feast 0.64.

En feast 0.64 el build de la UI (React) pide el registry como PROTOBUF en
``GET <registryPath>/registry`` (=> /api/v1/registry), pero el servidor de
``feast ui`` ya no expone esa ruta tras la refactorizacion a la API REST
(/api/v1/projects, /api/v1/entities, ...). Resultado: la UI carga en blanco.

Este script construye la app oficial (``feast.ui_server.get_app``) y le agrega
la ruta que falta, devolviendo el proto del registry completo. Como el registry
es compartido (volumen /feast_data), contiene los proyectos ``module1_features``
y ``hello_features`` y la UI los muestra ambos en su selector de proyectos.

Uso:  python serve_ui.py <repo_path> <host> <port>
"""

import importlib.resources as importlib_resources
import json
import sys

import uvicorn
from starlette.responses import Response
from starlette.routing import Mount

import feast.ui_server as ui_server
from feast import FeatureStore

repo_path, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])

store = FeatureStore(repo_path=repo_path)
app = ui_server.get_app(store, project_id=store.project, root_path="")

# get_app escribe projects-list.json con registryPath="/api/v1", pero el build
# de React usa esa URL LITERAL para pedir el proto del registry (y /api/v1 solo
# es el mount de la API REST -> 404 -> la UI muestra 0 objetos en todo proyecto).
# La reescribimos para que apunte a la ruta /registry que agregamos abajo.
with importlib_resources.as_file(
    importlib_resources.files("feast") / "ui/build/projects-list.json"
) as projects_list_path:
    projects_list = json.loads(projects_list_path.read_text())
    for entry in projects_list["projects"]:
        entry["registryPath"] = "/api/v1/registry"
    projects_list_path.write_text(json.dumps(projects_list))

# La sub-app REST esta montada en /api/v1; le agregamos la ruta /registry.
rest_app = next(
    r.app for r in app.routes
    if isinstance(r, Mount) and r.path == "/api/v1" and hasattr(r.app, "add_api_route")
)


def read_registry(allow_cache: bool = True, project: str = ""):
    # Refresca en cada peticion: los DAGs de Airflow aplican cambios al registry
    # compartido y la UI debe verlos sin reiniciar el contenedor.
    store.refresh_registry()
    proto = store.registry.proto()
    return Response(content=proto.SerializeToString(), media_type="application/octet-stream")


rest_app.add_api_route("/registry", read_registry, methods=["GET"])

print(f">> Feast UI (todos los proyectos del registry) en http://localhost:{port}")
uvicorn.run(app, host=host, port=port)
