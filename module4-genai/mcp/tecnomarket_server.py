"""Servidor MCP de TecnoMarket — expone el RAG del Módulo 4 por protocolo.

Generado desde notebooks/03_mcp_rag_agentes.ipynb. Ejecutar directo:
    uv run python mcp/tecnomarket_server.py     (transporte stdio)
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

MODULO = Path(__file__).resolve().parent.parent
load_dotenv(MODULO / ".env")

from google import genai                      # noqa: E402
from google.genai import types as gtypes     # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "tecnomarket")

mcp = FastMCP("tecnomarket")
_gclient = genai.Client()
_qdrant = QdrantClient(url=QDRANT_URL)


def _embed_consulta(texto: str) -> list[float]:
    r = _gclient.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=f"task: search result | query: {texto}",
        config=gtypes.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return list(r.embeddings[0].values)


@mcp.tool()
def buscar_catalogo(consulta: str, top_k: int = 4) -> str:
    """Busca en la base de conocimiento de TecnoMarket (políticas de envíos,
    devoluciones y garantías, y catálogo de productos) los pasajes más
    relevantes para una consulta en lenguaje natural."""
    hits = _qdrant.query_points(
        collection_name=COLLECTION, query=_embed_consulta(consulta),
        limit=top_k, with_payload=True,
    ).points
    bloques = []
    for i, h in enumerate(hits, 1):
        p = h.payload
        if p["tipo"] == "producto_imagen":
            bloques.append(f"[{i}] foto del producto {p['sku']} — {p['nombre']}")
        else:
            origen = p.get("fuente") or f"{p['sku']} — {p['nombre']}"
            bloques.append(f"[{i}] (fuente: {origen})\n{p['texto']}")
    return "\n\n".join(bloques) if bloques else "Sin resultados."


@mcp.tool()
def detalle_producto(sku: str) -> str:
    """Devuelve la ficha exacta de un producto de TecnoMarket dado su SKU
    (formato TM-XXXX)."""
    hits, _ = _qdrant.scroll(
        collection_name=COLLECTION, limit=1, with_payload=True,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="sku",
                                  match=models.MatchValue(value=sku.strip().upper())),
            models.FieldCondition(key="tipo",
                                  match=models.MatchValue(value="producto_texto")),
        ]),
    )
    if not hits:
        return f"No existe el SKU {sku}."
    p = hits[0].payload
    return json.dumps({"sku": p["sku"], "nombre": p["nombre"],
                       "descripcion": p["texto"]}, ensure_ascii=False)


@mcp.resource("politica://{nombre}")
def politica(nombre: str) -> str:
    """Texto completo de una política de TecnoMarket: envios, devoluciones o
    garantias."""
    ruta = MODULO / "rag" / "catalog" / "docs" / f"{nombre}.md"
    if not ruta.exists():
        return f"No existe la política '{nombre}'."
    return ruta.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(transport="stdio")
