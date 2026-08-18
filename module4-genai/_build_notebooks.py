"""Genera los notebooks del Módulo 4 con nbformat.

NO edites el JSON de los .ipynb a mano — ejecuta este script:

    uv run python _build_notebooks.py

Escribe (serie incremental):
    notebooks/01_rag_multimodal.ipynb      RAG multimodal (Qdrant + Gemini + Ollama cloud)
    notebooks/02_agentes_langgraph.ipynb   agente LangGraph con el RAG como herramienta
    notebooks/03_mcp_rag_agentes.ipynb     RAG + agentes + MCP conectados

Los notebooks NO se ejecutan previamente (no tienen outputs). Valida con:
    uv run python -c "import nbformat; nbformat.read('notebooks/01_rag_multimodal.ipynb', as_version=4); print('OK')"
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def write_notebook(cells: list[nbf.NotebookNode], path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    print(f"Escrito {path}")


# ===========================================================================
# Celdas compartidas (aparecen, en versión compacta, en los notebooks 02 y 03)
# ===========================================================================

CONFIG_CELL = r"""
# Carga la configuración desde module4-genai/.env (cópiala de .env.example).
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path.cwd()
if not (ROOT / "rag").exists():          # si ejecutas desde notebooks/
    ROOT = ROOT.parent
load_dotenv(ROOT / ".env")

GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "tecnomarket")

print("Módulo:", ROOT)
print("Embeddings:", GEMINI_EMBEDDING_MODEL, f"({EMBEDDING_DIM} dim)")
print("Qdrant:", QDRANT_URL, "| colección:", COLLECTION)
print("GEMINI_API_KEY definida:", bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")))
print("OLLAMA_API_KEY definida:", bool(os.getenv("OLLAMA_API_KEY")))
"""

LLM_CELL = r"""
# ⚙️ El modelo de Ollama cloud se elige AQUÍ, en el notebook.
#    Catálogo: https://ollama.com/search?c=cloud . Algunas opciones:
#      "minimax-m3:cloud"    razonador (thinking)
#      "kimi-k3:cloud"       multimodal (visión); se factura como "extra usage"
#      "gpt-oss:120b-cloud"  incluido en el plan gratuito
OLLAMA_MODEL = "minimax-m3:cloud"

# Si el modelo elegido no está disponible en tu plan (p. ej. HTTP 402 por saldo
# de extra usage en cero), caemos automáticamente al fallback del plan gratuito.
OLLAMA_FALLBACK = "gpt-oss:120b-cloud"

from langchain_ollama import ChatOllama
from ollama import Client as OllamaClient

OLLAMA_HEADERS = {"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]}

def conectar_llm(**kwargs):
    # kwargs extra van directo a ChatOllama (p. ej. reasoning=True).
    probe = OllamaClient(host=OLLAMA_HOST, headers=OLLAMA_HEADERS)
    for modelo in [OLLAMA_MODEL, OLLAMA_FALLBACK]:
        try:
            probe.chat(model=modelo,
                       messages=[{"role": "user", "content": "ok"}],
                       options={"num_predict": 1})
        except Exception as exc:
            print(f"⚠ {modelo} no disponible: {str(exc)[:110]}")
            continue
        print("✔ Usando el modelo:", modelo)
        return ChatOllama(
            model=modelo,
            base_url=OLLAMA_HOST,
            client_kwargs={"headers": OLLAMA_HEADERS},
            temperature=0.1,
            **kwargs,
        )
    raise RuntimeError("Ningún modelo de Ollama cloud respondió; revisa OLLAMA_API_KEY.")

llm = conectar_llm()
"""


# ===========================================================================
# Notebook 1 — RAG multimodal
# ===========================================================================
def build_notebook_1() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# 01 — RAG multimodal: Qdrant + Gemini Embeddings + LLM en Ollama cloud

Primer notebook de una serie **incremental** de tres:

| Notebook | Qué construye |
|---|---|
| **01 (este)** | Un pipeline de **RAG multimodal** completo y evaluado |
| 02 | Un **agente** LangGraph que usa este RAG como *herramienta* |
| 03 | Los mismos componentes conectados vía **MCP** |

## El caso de uso

**TecnoMarket**, una tienda en línea ficticia, quiere un asistente que responda
preguntas usando su conocimiento interno:

- **Documentos de políticas** (envíos, devoluciones, garantías) — texto largo
  que hay que *trocear*.
- **Catálogo de productos** — cada producto tiene descripción **y fotografía**.

La gracia del caso es que es **multimodal**: el modelo de embeddings
`gemini-embedding-2` proyecta **texto e imágenes al MISMO espacio vectorial**,
así que una consulta escrita ("zapatillas de lona rojas") puede recuperar
directamente la *foto* del producto, sin pasos intermedios de anotación manual.

## La arquitectura

```
 políticas (.md) ──▶ chunking ──▶ embed (texto) ──┐
 descripciones  ─────────────────▶ embed (texto) ──┤──▶ ┌────────┐
 fotos (.jpg)   ─────────────────▶ embed (imagen) ─┘    │ Qdrant │  vectores + payloads
                                                        └───┬────┘
 pregunta ──▶ embed ──▶ búsqueda densa ─┐                   │
          └─▶ BM25 (léxica) ────────────┤──▶ fusión RRF ◀───┘
                                        ▼
                            contexto recuperado (texto + imágenes)
                                        ▼
                       prompt (LangChain) ──▶ LLM (Ollama cloud) ──▶ respuesta + fuentes
```

## Lo que cubrimos

1. **Embeddings multimodales** — un solo espacio para texto e imagen (Gemini).
2. **Qdrant** — desplegar la base vectorial y diseñar la colección.
3. **Chunking** — estrategia de troceado con LangChain.
4. **Ingesta** — indexar chunks, descripciones e imágenes.
5. **Recuperación** — densa, léxica (BM25) e **híbrida** (RRF).
6. **Generación** — cadena de LangChain con un LLM de Ollama cloud.
7. **Evaluación** — *hit rate* y *MRR*, explicados y medidos.
8. **Widgets** — explorar y validar los resultados interactivamente.

## Antes de ejecutar

```bash
cd module4-genai
uv sync                                   # dependencias
cp .env.example .env                      # y rellena GEMINI_API_KEY y OLLAMA_API_KEY
cd qdrant && docker compose up -d && cd . # desplegar Qdrant
```

- `GEMINI_API_KEY`: https://aistudio.google.com/apikey (los embeddings usan la API de Gemini).
- `OLLAMA_API_KEY`: https://ollama.com/settings/keys (el LLM corre en Ollama cloud).
- Dashboard de Qdrant: <http://localhost:6333/dashboard>
"""))

    cells.append(code(CONFIG_CELL))

    # --- Caso de uso: cargar el catálogo ---
    cells.append(md(r"""
## 1. Los datos: catálogo y políticas de TecnoMarket

Los datos viven en `rag/catalog/`:

```
rag/catalog/
├── products.json    ~100 productos: sku, nombre, categoría, precio, descripción, imagen
├── docs/            3 políticas en Markdown: envíos, devoluciones, garantías
└── images/          ~100 fotografías de producto (JPEG, licencias CC)
```

Con ~100 productos en ~26 categorías el índice deja de ser de juguete: hay
**distractores** (varias zapatillas, varios relojes…), que es lo que hace
interesante medir la calidad de la recuperación en la sección 9.
"""))

    cells.append(code(r"""
import base64
import json

import pandas as pd

CATALOG_DIR = ROOT / "rag" / "catalog"
PRODUCTS = json.loads((CATALOG_DIR / "products.json").read_text(encoding="utf-8"))
DOCS = {p.name: p.read_text(encoding="utf-8")
        for p in sorted((CATALOG_DIR / "docs").glob("*.md"))}

df_prod = pd.DataFrame(PRODUCTS)
print(f"{len(PRODUCTS)} productos en {df_prod['categoria'].nunique()} categorías, "
      f"{len(DOCS)} documentos de políticas: {list(DOCS)}")
print("\nProductos por categoría:")
print(df_prod["categoria"].value_counts().to_string())
df_prod[["sku", "nombre", "categoria", "precio", "imagen"]].head(8)
"""))

    cells.append(code(r"""
# Mini-galería del catálogo. Incrustamos las imágenes como base64 para que el
# HTML sea autocontenido (mismo truco que usaremos al mostrar resultados).
from IPython.display import HTML, display

def img_b64(path):
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()

def galeria(productos, ancho=140):
    tarjetas = []
    for p in productos:
        src = img_b64(CATALOG_DIR / "images" / p["imagen"])
        tarjetas.append(
            f'<div style="display:inline-block;margin:6px;text-align:center;width:{ancho}px;'
            f'vertical-align:top;font-family:sans-serif;font-size:11px">'
            f'<img src="{src}" style="width:{ancho}px;height:{ancho}px;object-fit:cover;'
            f'border-radius:8px"><br><b>{p["sku"]}</b><br>{p["nombre"]}</div>'
        )
    return HTML("<div>" + "".join(tarjetas) + "</div>")

display(galeria(PRODUCTS[:12]))
print(f"(mostrando 12 de {len(PRODUCTS)}; el resto se explora con los widgets de la sección 10)")
"""))

    # --- Embeddings multimodales ---
    cells.append(md(r"""
## 2. Embeddings multimodales con `gemini-embedding-2`

Un **embedding** convierte un contenido en un vector denso en $\mathbb{R}^d$,
de modo que contenidos semánticamente parecidos queden **cerca**. La medida de
cercanía que usamos es la **similitud coseno**:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a}\rVert\,\lVert \mathbf{b}\rVert}$$

Lo nuevo aquí: `gemini-embedding-2` es **nativamente multimodal**. Texto,
imágenes, audio y PDF se proyectan al **mismo espacio vectorial**, así que
podemos comparar directamente una *consulta escrita* contra la *foto* de un
producto. Antes de estos modelos, el truco habitual era generar una descripción
de cada imagen con un modelo de visión y embeber esa descripción — un paso
extra, con pérdida de información.

Tres detalles prácticos del modelo:

- **Dimensiones flexibles (MRL)** — *Matryoshka Representation Learning*
  permite truncar el vector de 3072 a 1536 o **768** dims (nuestro caso: 4x
  menos almacenamiento con una pérdida de calidad mínima). El modelo
  **re-normaliza automáticamente** el vector truncado (norma 1).
- **Formato de la entrada** — para recuperación, la convención del modelo es
  formatear consulta y documento de forma distinta (asimétrica):
  - consulta: `task: search result | query: {texto}`
  - documento: `title: {título} | text: {texto}`
- **⚠ Agregación** — si pasas una **lista** en `contents`, el modelo devuelve
  **UN solo embedding agregado** de todo el conjunto (útil para "texto +
  imagen = un producto", traicionero si esperabas un embedding por elemento).
  Por eso embebemos **elemento por elemento**.
"""))

    cells.append(code(r"""
import time
from functools import lru_cache

from google import genai
from google.genai import types

gclient = genai.Client()  # lee GEMINI_API_KEY / GOOGLE_API_KEY del entorno

def _embed(contents):
    # Reintentos con backoff exponencial: la API de Gemini tiene límites de
    # tasa (429) que en el tier gratuito se alcanzan rápido.
    for intento in range(5):
        try:
            r = gclient.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
            )
            return list(r.embeddings[0].values)
        except Exception as exc:
            espera = 2 ** intento
            print(f"  reintento en {espera}s ({str(exc)[:80]})")
            time.sleep(espera)
    raise RuntimeError("La API de embeddings no respondió tras 5 intentos.")

@lru_cache(maxsize=512)
def embed_consulta(texto):
    # Cache en memoria: la evaluación (§9) y los widgets (§10) repiten las
    # mismas consultas — no tiene sentido pagar la API dos veces por ellas.
    return _embed(f"task: search result | query: {texto}")

def embed_documento(texto, titulo="none"):
    return _embed(f"title: {titulo} | text: {texto}")

def embed_imagen(path):
    part = types.Part.from_bytes(data=Path(path).read_bytes(), mime_type="image/jpeg")
    return _embed([part])

v = embed_consulta("zapatillas de lona rojas y azules")
print("Dimensión:", len(v), "| primeros 5 valores:", [round(x, 4) for x in v[:5]])
"""))

    cells.append(code(r"""
# La prueba de fuego multimodal: consultas de TEXTO contra FOTOS de producto.
# Sin OCR, sin descripciones intermedias: coseno directo entre modalidades.
import numpy as np

consultas = [
    "zapatillas de lona rojas y azules",
    "una máquina para preparar café espresso",
    "bicicleta de montaña con doble suspensión",
]
imagenes = ["zapatillas-urbanas.jpg", "cafetera-espresso.jpg",
            "bicicleta-montana.jpg", "teclado-mecanico.jpg"]

vecs_img = {img: np.array(embed_imagen(CATALOG_DIR / "images" / img))
            for img in imagenes}

filas = []
for q in consultas:
    vq = np.array(embed_consulta(q))
    filas.append({img: round(float(vq @ v / (np.linalg.norm(vq) * np.linalg.norm(v))), 3)
                  for img, v in vecs_img.items()})

pd.DataFrame(filas, index=consultas)
# Cada fila debería tener su máximo en la imagen correcta.
"""))

    # --- Qdrant ---
    cells.append(md(r"""
## 3. Desplegar Qdrant y diseñar la colección

**Qdrant** es nuestra base de datos vectorial. La desplegamos con Docker
(`qdrant/docker-compose.yml` de este módulo):

```bash
cd qdrant && docker compose up -d
```

Comparar la consulta contra cada vector almacenado sería $O(N)$; Qdrant indexa
con **HNSW** (*Hierarchical Navigable Small World*), un grafo multicapa que da
búsquedas de vecinos casi logarítmicas a cambio de una pérdida mínima de
*recall* (por eso "ANN": *Approximate Nearest Neighbors*).

Diseño de nuestra colección:

- **Vectores**: 768 dims, distancia **coseno**.
- **Payload** (metadatos JSON por punto) — la clave del diseño multimodal es
  el campo `tipo`, que nos deja filtrar por modalidad:

| campo | valores | para qué |
|---|---|---|
| `tipo` | `doc` / `producto_texto` / `producto_imagen` | filtrar por modalidad |
| `ref` | `envios.md#2`, `TM-1001:texto`, `TM-1001:imagen` | identificar el punto (evaluación) |
| `texto` | el chunk o la descripción (vacío en imágenes) | armar el contexto del LLM |
| `sku`, `nombre`, `imagen`, `fuente` | metadatos del catálogo | mostrar fuentes / filtrar |
"""))

    cells.append(code(r"""
from qdrant_client import QdrantClient, models

qdrant = QdrantClient(url=QDRANT_URL)
try:
    qdrant.get_collections()
except Exception as exc:
    raise RuntimeError(
        f"Qdrant no responde en {QDRANT_URL}. "
        "Inícialo con: cd qdrant && docker compose up -d"
    ) from exc

if qdrant.collection_exists(COLLECTION):
    qdrant.delete_collection(COLLECTION)
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=EMBEDDING_DIM,
                                       distance=models.Distance.COSINE),
)
print(f"Colección '{COLLECTION}' creada ({EMBEDDING_DIM} dim, coseno).")
"""))

    # --- Chunking ---
    cells.append(md(r"""
## 4. Estrategia de chunking (troceado) con LangChain

Los documentos largos no se embeben enteros. Dos razones:

1. **Precisión de la recuperación** — un vector que promedia 5 temas no queda
   cerca de ninguno. Chunks enfocados = vectores enfocados.
2. **Contexto del LLM** — recuperamos pasajes, no archivos: el prompt final
   debe ser pequeño y relevante.

Las decisiones de la estrategia:

- **Tamaño del chunk** — chico (200–400 chars): precisión alta, pero
  fragmentos sin contexto suficiente para el LLM. Grande (1500+): más
  contexto, pero vectores difusos y prompts caros. Para políticas cortas como
  las nuestras, un punto medio de **~700 caracteres** funciona bien.
- **Solapamiento (overlap)** — repetir el final de un chunk al inicio del
  siguiente (~15–20%) evita perder las ideas que cruzan la frontera.
- **Dónde cortar** — un corte a mitad de frase rompe el significado. El
  `RecursiveCharacterTextSplitter` de LangChain intenta separadores en orden
  (`\n\n` párrafos → `\n` líneas → `. ` frases → espacios) y solo baja al
  siguiente nivel si el fragmento sigue siendo demasiado grande.

Nuestros documentos además tienen **estructura Markdown** (secciones `##`), y
los límites de sección son límites semánticos naturales: cortar primero por
párrafos los respeta bastante bien.
"""))

    cells.append(code(r"""
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

texto_envios = DOCS["envios.md"]

# Ingenuo: corta cada N caracteres, sin mirar dónde.
naive = CharacterTextSplitter(separator="", chunk_size=700, chunk_overlap=0)
# Recursivo: respeta párrafos > líneas > frases > palabras, con solapamiento.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)

print("=== Corte ingenuo (fin del chunk 1) ===")
print("…" + naive.split_text(texto_envios)[0][-80:])
print("\n=== Corte recursivo (fin del chunk 1) ===")
print("…" + splitter.split_text(texto_envios)[0][-80:])
"""))

    cells.append(code(r"""
# Troceamos los tres documentos de políticas. Cada chunk conserva su fuente y
# su posición: la referencia "envios.md#2" identifica al chunk 2 de envios.md.
CHUNKS = []
for fuente, texto in DOCS.items():
    for i, chunk in enumerate(splitter.split_text(texto)):
        CHUNKS.append({"ref": f"{fuente}#{i}", "fuente": fuente, "texto": chunk})

print(f"{len(CHUNKS)} chunks en total")
pd.DataFrame([{"ref": c["ref"], "caracteres": len(c["texto"]),
               "inicio": c["texto"][:60].replace("\n", " ") + "…"} for c in CHUNKS])
"""))

    # --- Ingesta ---
    cells.append(md(r"""
## 5. Ingesta multimodal

Indexamos **tres tipos de puntos** en la misma colección:

1. los **chunks** de las políticas (embedding de texto),
2. las **descripciones** de los productos (embedding de texto),
3. las **fotografías** de los productos (embedding de imagen).

Cada punto lleva un ID **determinista** (UUID5 derivado de su `ref`): si
re-ejecutas la ingesta, los puntos se sobreescriben en lugar de duplicarse.

> **Cache de embeddings en disco.** Con ~100 productos, la ingesta completa
> son **~215 llamadas** a la API (≈15 chunks + 100 descripciones + 100 fotos)
> — varios minutos con los límites del tier gratuito. Para no pagar eso en
> cada re-ejecución, cacheamos cada vector en
> `rag/catalog/embeddings_cache.npz`, **con clave = hash del contenido**: si
> el contenido no cambió, el vector sale del disco; si editas una descripción
> o cambias el modelo/dimensión, solo ese elemento vuelve a la API. El repo
> incluye el cache ya calculado, así que la primera ingesta también es rápida.
"""))

    cells.append(code(r"""
import hashlib
import uuid

CACHE_PATH = CATALOG_DIR / "embeddings_cache.npz"
_cache = {}
if CACHE_PATH.exists():
    with np.load(CACHE_PATH) as z:
        _cache = {k: z[k] for k in z.files}
print(f"Cache de embeddings: {len(_cache)} vectores en disco")

_cache_nuevos = 0

def _clave(*partes):
    return hashlib.sha1("|".join(partes).encode()).hexdigest()

def _cacheado(clave, calcular):
    # Devuelve el vector del cache, o lo calcula (API) y lo agrega.
    global _cache_nuevos
    if clave not in _cache:
        _cache[clave] = np.asarray(calcular(), dtype=np.float32)
        _cache_nuevos += 1
    return _cache[clave].tolist()

def punto(ref, vector, payload):
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, ref)),
        vector=vector,
        payload={"ref": ref, **payload},
    )

BASE = (GEMINI_EMBEDDING_MODEL, str(EMBEDDING_DIM))
puntos = []

for c in CHUNKS:  # 1) chunks de políticas
    vec = _cacheado(_clave("doc", *BASE, c["fuente"], c["texto"]),
                    lambda c=c: embed_documento(c["texto"], titulo=c["fuente"]))
    puntos.append(punto(c["ref"], vec,
                        {"tipo": "doc", "fuente": c["fuente"], "texto": c["texto"]}))
print(f"✔ {len(CHUNKS)} chunks de políticas")

for i, p in enumerate(PRODUCTS):  # 2) descripciones + 3) fotografías
    vec_txt = _cacheado(_clave("texto", *BASE, p["nombre"], p["descripcion"]),
                        lambda p=p: embed_documento(p["descripcion"], titulo=p["nombre"]))
    puntos.append(punto(f"{p['sku']}:texto", vec_txt,
                        {"tipo": "producto_texto", "sku": p["sku"], "nombre": p["nombre"],
                         "imagen": p["imagen"], "texto": p["descripcion"]}))
    ruta_img = CATALOG_DIR / "images" / p["imagen"]
    vec_img = _cacheado(_clave("imagen", *BASE, hashlib.sha1(ruta_img.read_bytes()).hexdigest()),
                        lambda r=ruta_img: embed_imagen(r))
    puntos.append(punto(f"{p['sku']}:imagen", vec_img,
                        {"tipo": "producto_imagen", "sku": p["sku"], "nombre": p["nombre"],
                         "imagen": p["imagen"], "texto": ""}))
    if (i + 1) % 20 == 0:
        print(f"  … {i + 1}/{len(PRODUCTS)} productos")
print(f"✔ {len(PRODUCTS)} descripciones y {len(PRODUCTS)} imágenes")

if _cache_nuevos:
    np.savez_compressed(CACHE_PATH, **_cache)
    print(f"Cache actualizado: +{_cache_nuevos} vectores nuevos → {CACHE_PATH.name}")
else:
    print("Todo salió del cache: 0 llamadas a la API en la ingesta.")

qdrant.upsert(collection_name=COLLECTION, points=puntos)
print(f"\nInsertados {len(puntos)} puntos. Total en Qdrant:",
      qdrant.count(COLLECTION).count)

# Índices auxiliares en memoria que reutilizaremos:
PAYLOAD_POR_REF = {pt.payload["ref"]: pt.payload for pt in puntos}
CORPUS_TEXTO = [(ref, pl["texto"]) for ref, pl in PAYLOAD_POR_REF.items() if pl["texto"]]
"""))

    # --- Recuperación densa ---
    cells.append(md(r"""
## 6. Recuperación densa (semántica)

Embebemos la consulta (con su formato `task: search result | query: …`) y
pedimos a Qdrant los `top_k` vecinos por coseno. Gracias al campo `tipo` del
payload también podemos **filtrar por modalidad**.
"""))

    cells.append(code(r"""
def buscar_por_vector(vector, top_k=5, tipo=None):
    # Núcleo de la búsqueda densa: recibe un VECTOR (da igual si vino de un
    # texto o de una imagen — mismo espacio) y consulta Qdrant.
    filtro = None
    if tipo is not None:
        filtro = models.Filter(must=[models.FieldCondition(
            key="tipo", match=models.MatchValue(value=tipo))])
    hits = qdrant.query_points(
        collection_name=COLLECTION, query=vector,
        limit=top_k, with_payload=True, query_filter=filtro,
    ).points
    return [{"score": round(h.score, 3), **h.payload} for h in hits]

def buscar_denso(consulta, top_k=5, tipo=None):
    return buscar_por_vector(embed_consulta(consulta), top_k=top_k, tipo=tipo)

def mostrar(resultados):
    piezas = []
    for i, r in enumerate(resultados, 1):
        cabeza = f'<b>[{i}]</b> <code>{r["ref"]}</code> · score {r["score"]} · <i>{r["tipo"]}</i>'
        cuerpo = (r["texto"][:180] + "…") if r["texto"] else f'<b>{r.get("nombre", "")}</b> (imagen)'
        thumb = ""
        if r.get("imagen"):
            thumb = (f'<img src="{img_b64(CATALOG_DIR / "images" / r["imagen"])}" '
                     f'style="width:72px;height:72px;object-fit:cover;border-radius:6px;'
                     f'margin-right:10px;float:left">')
        piezas.append(f'<div style="overflow:auto;margin:6px 0;font-family:sans-serif;'
                      f'font-size:12px">{thumb}{cabeza}<br>{cuerpo}</div>')
    return HTML("".join(piezas))

mostrar(buscar_denso("¿cuánto tarda el envío express?", top_k=3))
"""))

    cells.append(code(r"""
# Consulta cross-modal: pedimos SOLO imágenes. El texto de la consulta se
# compara directamente contra los vectores de las fotos.
mostrar(buscar_denso("algo para escuchar música sin ruido de fondo",
                     top_k=3, tipo="producto_imagen"))
"""))

    # --- Híbrida ---
    cells.append(md(r"""
## 7. Búsqueda híbrida: densa + BM25, fusionadas con RRF

La búsqueda densa entiende **paráfrasis** ("algo para escuchar música" →
audífonos), pero puede tropezar con **términos exactos** poco frecuentes:
SKUs, códigos, nombres propios. Ahí brilla la búsqueda **léxica** clásica.

**BM25** puntúa un documento $d$ para una consulta $q$ sumando, por cada
término $t$ de la consulta:

$$\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t)\cdot
\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\overline{|d|}}\right)}$$

donde $f(t,d)$ es la frecuencia del término en el documento, $\text{IDF}$
castiga los términos comunes, y $k_1, b$ son constantes. La intuición: premia
coincidencias exactas de palabras raras, con saturación.

Para **fusionar** ambos rankings usamos *Reciprocal Rank Fusion* (RRF), que
combina **posiciones** (no puntuaciones, que viven en escalas incomparables):

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}$$

con $k = 60$ típicamente (amortigua el peso de los primeros puestos).

> Detalle multimodal honesto: las **imágenes no tienen texto**, así que BM25
> no puede verlas — solo la rama densa las recupera. La fusión igual funciona:
> los puntos que solo aparecen en un ranking simplemente suman un término.
"""))

    cells.append(code(r"""
from rank_bm25 import BM25Okapi

def _tokenizar(texto):
    return texto.lower().split()

_refs_bm25 = [ref for ref, _ in CORPUS_TEXTO]
_bm25 = BM25Okapi([_tokenizar(texto) for _, texto in CORPUS_TEXTO])

def buscar_bm25(consulta, top_k=5):
    scores = _bm25.get_scores(_tokenizar(consulta))
    orden = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"score": round(float(scores[i]), 3), **PAYLOAD_POR_REF[_refs_bm25[i]]}
            for i in orden if scores[i] > 0]

def buscar_hibrido(consulta, top_k=5, pool=15, rrf_k=60):
    denso = buscar_denso(consulta, top_k=pool)
    lexico = buscar_bm25(consulta, top_k=pool)
    fusion = {}
    for rank, r in enumerate(denso):
        fusion[r["ref"]] = fusion.get(r["ref"], 0) + 1.0 / (rrf_k + rank)
    for rank, r in enumerate(lexico):
        fusion[r["ref"]] = fusion.get(r["ref"], 0) + 1.0 / (rrf_k + rank)
    mejores = sorted(fusion, key=fusion.get, reverse=True)[:top_k]
    return [{"score": round(fusion[ref], 5), **PAYLOAD_POR_REF[ref]} for ref in mejores]

# El caso donde lo léxico gana: una consulta con SKU exacto.
q = "características de la bicicleta TM-1006"
print("— Solo denso:")
for r in buscar_denso(q, top_k=3):
    print(f"   [{r['score']}] {r['ref']}")
print("— Solo BM25:")
for r in buscar_bm25(q, top_k=3):
    print(f"   [{r['score']}] {r['ref']}")
print("— Híbrido (RRF):")
for r in buscar_hibrido(q, top_k=3):
    print(f"   [{r['score']}] {r['ref']}")
"""))

    # --- Generación ---
    cells.append(md(r"""
## 8. Generación: LangChain + LLM en Ollama cloud

Para la generación usamos un LLM servido por **Ollama cloud** — mismo
protocolo que un Ollama local, pero el modelo (de cientos de miles de millones
de parámetros) corre en los servidores de Ollama y nos autenticamos con una
API key.

> **El modelo se elige en la celda siguiente** (variable `OLLAMA_MODEL`), no
> en el `.env`: cambiarlo es parte del experimento. Ojo con la facturación:
> algunos modelos (p. ej. `kimi-k3:cloud`) son *extra usage* y requieren plan
> de pago con saldo; si el elegido no responde, la celda **degrada
> automáticamente** a `gpt-oss:120b-cloud` (plan gratuito). El pipeline es
> idéntico con cualquiera.

La integración con LangChain es `ChatOllama` (paquete `langchain-ollama`)
apuntando `base_url` a `https://ollama.com` con el header de autorización.
"""))

    cells.append(code(LLM_CELL))

    cells.append(md(r"""
### La cadena de RAG con LCEL

Componemos el pipeline con la sintaxis declarativa de LangChain (*LCEL*):

```
{contexto: recuperar, pregunta: passthrough} → prompt → llm → parser
```

El prompt instruye al modelo a responder **solo** con el contexto recuperado y
a citar las fuentes `[n]` — el mecanismo anti-alucinación básico de RAG.
"""))

    cells.append(code(r"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

PROMPT_RAG = ChatPromptTemplate.from_messages([
    ("system",
     "Eres el asistente de la tienda TecnoMarket. Responde en español usando "
     "ÚNICAMENTE el contexto proporcionado. Cita las fuentes con su número "
     "entre corchetes, como [1]. Si el contexto no contiene la respuesta, "
     "dilo claramente y no inventes."),
    ("human", "Contexto:\n{contexto}\n\nPregunta: {pregunta}"),
])

def formatear_contexto(resultados):
    bloques = []
    for i, r in enumerate(resultados, 1):
        if r["tipo"] == "producto_imagen":
            bloques.append(f"[{i}] (foto del producto {r['sku']} — {r['nombre']})")
        else:
            origen = r.get("fuente") or f"{r['sku']} — {r['nombre']}"
            bloques.append(f"[{i}] (fuente: {origen})\n{r['texto']}")
    return "\n\n".join(bloques)

def recuperar(pregunta, top_k=4):
    return buscar_hibrido(pregunta, top_k=top_k)

cadena_rag = (
    {"contexto": lambda x: formatear_contexto(recuperar(x)),
     "pregunta": RunnablePassthrough()}
    | PROMPT_RAG
    | llm
    | StrOutputParser()
)

pregunta = "¿Puedo devolver unos audífonos si ya abrí el empaque?"
print(cadena_rag.invoke(pregunta))
"""))

    cells.append(code(r"""
# La misma respuesta, pero mostrando también las fuentes recuperadas (con foto
# cuando la fuente es una imagen del catálogo) — así se audita un RAG.
def responder_con_fuentes(pregunta, top_k=4):
    fuentes = recuperar(pregunta, top_k=top_k)
    respuesta = (PROMPT_RAG | llm | StrOutputParser()).invoke(
        {"contexto": formatear_contexto(fuentes), "pregunta": pregunta})
    print(respuesta)
    display(mostrar(fuentes))

responder_con_fuentes("¿Qué garantía tiene la cafetera y qué no cubre?")
"""))

    cells.append(md(r"""
### Bonus multimodal: el LLM también puede *ver* la imagen recuperada

Si el modelo elegido tiene **visión nativa** (p. ej. `kimi-k3:cloud`), además
de citar la foto como fuente podemos adjuntarla al mensaje para que la
describa o la use al responder. (Con un modelo solo-texto — `minimax-m3`,
`gpt-oss` — la celda lo detecta y lo explica en lugar de fallar.)
"""))

    cells.append(code(r"""
consulta_visual = "zapatillas de lona para uso diario"
foto = next(r for r in buscar_denso(consulta_visual, top_k=1, tipo="producto_imagen"))
print("Imagen recuperada:", foto["ref"], "—", foto["nombre"])

try:
    cliente_ollama = OllamaClient(host=OLLAMA_HOST, headers=OLLAMA_HEADERS)
    r = cliente_ollama.chat(
        model=llm.model,
        messages=[{
            "role": "user",
            "content": "Describe esta foto de producto en una frase de catálogo, en español.",
            "images": [base64.b64encode(
                (CATALOG_DIR / "images" / foto["imagen"]).read_bytes()).decode()],
        }],
    )
    print("\n", r["message"]["content"])
except Exception as exc:
    print(f"\n(El modelo '{llm.model}' no aceptó la imagen: {str(exc)[:100]}.\n"
          " Con un modelo de visión nativa (p. ej. kimi-k3:cloud) esta celda "
          "describe la foto.)")
"""))

    # --- Evaluación ---
    cells.append(md(r"""
## 9. Evaluar la recuperación: *hit rate* y *MRR*

En RAG, si la recuperación falla, la generación no puede salvarla: el LLM no
puede citar lo que nunca vio. Por eso la recuperación se evalúa **por
separado** del LLM, y con métricas propias.

### El vocabulario, pieza por pieza

- **Evalset**: una lista de consultas de prueba. Para cada una, anotamos de
  antemano qué elementos del índice **cuentan como respuesta correcta**. A esa
  anotación se le llama ***ground truth*** (la verdad de referencia).
- **Relevante**: un resultado que está en el ground truth de esa consulta. Es
  una etiqueta **binaria**: cuenta o no cuenta (no hay "medio relevante").
- **Ranking**: la lista ordenada que devuelve el buscador. La **posición**
  (o *rank*) se cuenta desde 1.
- **@k** ("*at* k"): solo miramos los **primeros $k$** resultados del ranking
  e ignoramos el resto. ¿Por qué truncar? Porque al LLM solo le pasamos los
  top-$k$ chunks — un relevante en la posición 47 **no existe** para el
  usuario ni para el prompt. La métrica debe mirar lo mismo que el sistema usa.

### ¿Qué cuenta como "relevante" en NUESTRO evalset?

Dos reglas, según el tipo de consulta:

1. **Consultas sobre políticas** ("¿cuánto cuesta el envío express?"):
   cualquier chunk del documento correcto es relevante — `envios.md#0`,
   `envios.md#1`, … Todos contienen contexto legítimo del tema.
2. **Consultas de producto** ("audífonos con cancelación de ruido"): con ~100
   productos ya no hay UNA respuesta correcta — cualquier audífono del
   catálogo sirve. Por eso marcamos como relevante **todo producto de la
   categoría correcta** (su `:texto` y su `:imagen`). Esto se llama relevancia
   a nivel de categoría.

### Hit Rate @ k — "¿encontró *algo* útil?"

Para **una** consulta: vale **1** si entre los primeros $k$ resultados hay al
menos un relevante, y **0** si no. Para el evalset completo, se promedia:

$$\text{HitRate@}k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}\left[\,\exists\, d \in \text{top-}k(q) : d \text{ es relevante}\,\right]$$

($\mathbb{1}[\cdot]$ vale 1 si la condición se cumple, 0 si no; $|Q|$ es el
número de consultas.) Un HitRate@5 de 0.80 se lee: *"en el 80% de las
consultas, el top-5 contenía al menos una respuesta útil"*. Es la métrica de
**cobertura**: ¿le dimos al LLM alguna oportunidad de responder bien?

Su punto ciego: **ignora la posición dentro del top-k**. Relevante en el
puesto 1 y relevante en el puesto 5 puntúan igual (1.0).

### MRR @ k — "¿qué tan *arriba* quedó?"

Primero, para **una** consulta, el ***reciprocal rank* (RR)**: el inverso de
la posición del **primer** relevante.

| posición del 1er relevante | RR = 1/posición |
|---|---|
| 1 | 1.00 |
| 2 | 0.50 |
| 3 | 0.33 |
| 5 | 0.20 |
| no aparece en el top-k | 0.00 |

El inverso hace que la penalización sea **no lineal**: caer del puesto 1 al 2
cuesta 0.5 puntos, pero del 4 al 5 apenas 0.05. Refleja cómo se consume un
ranking: lo primero pesa mucho más.

**MRR** (*Mean* Reciprocal Rank) es el promedio de los RR sobre el evalset:

$$\text{MRR@}k = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}
\qquad \text{(término} = 0 \text{ si no hay relevante en el top-}k\text{)}$$

Un MRR@5 de 0.70 se lee: *"en promedio, el primer resultado útil aparece
entre el puesto 1 y el 2"*. Es la métrica de **posición**.

### Ejemplo trabajado completo (3 consultas, k=3)

| consulta | top-3 (✓ = relevante) | hit@3 | rank 1er ✓ | RR@3 |
|---|---|---|---|---|
| A | **✓** ✗ ✗ | 1 | 1 | 1.00 |
| B | ✗ ✗ **✓** | 1 | 3 | 0.33 |
| C | ✗ ✗ ✗ | 0 | — | 0.00 |

$$\text{HitRate@3} = \frac{1+1+0}{3} = 0.67 \qquad
\text{MRR@3} = \frac{1.00 + 0.33 + 0.00}{3} = 0.44$$

**Por qué se necesitan las dos:** un sistema puede encontrar casi siempre algo
relevante (HitRate alto) pero enterrado en el puesto 4–5 (MRR bajo): "funciona,
pero el prompt del LLM arranca con ruido". Otro puede ser preciso cuando
acierta (MRR alto por sus unos) pero fallar en muchas consultas. Mirar solo
una métrica esconde uno de los dos problemas.

Empecemos por ver la **anatomía de una sola consulta**, con el ranking real y
sus marcas ✓/✗ — y luego sí, el agregado.
"""))

    cells.append(code(r"""
# Ground truth: relevancia por CATEGORÍA para productos, por DOCUMENTO para
# políticas. Un prefijo "TM-1042" marca TM-1042:texto y TM-1042:imagen;
# un prefijo "envios.md" marca envios.md#0, envios.md#1, …
def skus_de(categoria):
    return [p["sku"] for p in PRODUCTS if p["categoria"] == categoria]

def es_relevante(ref, prefijos):
    return any(ref.startswith(p) for p in prefijos)

def rank_primer_relevante(refs, prefijos):
    # Posición (1-indexada) del primer resultado relevante; None si no aparece.
    for i, ref in enumerate(refs, 1):
        if es_relevante(ref, prefijos):
            return i
    return None

# --- Anatomía de UNA consulta: el ranking crudo con sus marcas ---
consulta = "audífonos inalámbricos con cancelación de ruido"
prefijos = skus_de("audifonos")
print(f"Consulta:   {consulta!r}")
print(f"Relevantes: cualquier ref que empiece por {prefijos}\n")

k = 5
refs = [r["ref"] for r in buscar_hibrido(consulta, top_k=k)]
for pos, ref in enumerate(refs, 1):
    marca = "✓" if es_relevante(ref, prefijos) else "✗"
    print(f"  {pos}. {marca} {ref}")

rank = rank_primer_relevante(refs, prefijos)
print(f"\nhit@{k} = {1 if rank else 0}   (¿hay al menos un ✓ en el top-{k}?)")
print(f"rank del primer ✓ = {rank}   →   RR@{k} = " +
      (f"1/{rank} = {1/rank:.2f}" if rank else "0.00"))
print("\nEl evalset repite esto para todas las consultas y promedia.")
"""))

    cells.append(code(r"""
EVALSET = [
    # (consulta, prefijos de refs relevantes)
    # — políticas: relevante = cualquier chunk del documento correcto
    ("¿Cuánto cuesta el envío express y en cuánto tiempo llega?", ["envios.md"]),
    ("¿Qué pasa si la transportadora no logra entregar mi paquete?", ["envios.md"]),
    ("¿Puedo devolver unos audífonos si ya abrí la caja?", ["devoluciones.md"]),
    ("¿Cómo inicio una devolución desde mi cuenta?", ["devoluciones.md"]),
    ("¿Cuántos meses de garantía tiene el teclado mecánico?", ["garantias.md"]),
    ("¿La garantía cubre el desgaste de la batería?", ["garantias.md"]),
    # — término exacto: aquí el único relevante SÍ es un producto concreto
    ("bicicleta TM-1006 precio y características", ["TM-1006"]),
    # — productos: relevante = cualquier producto de la categoría correcta
    ("zapatillas para correr con buena amortiguación", skus_de("zapatillas")),
    ("audífonos inalámbricos con cancelación de ruido", skus_de("audifonos")),
    ("una máquina para preparar café espresso en casa", skus_de("cafeteras")),
    ("mochila para hacer senderismo de varios días", skus_de("mochilas")),
    ("reloj de pulsera analógico clásico", skus_de("relojes")),
    ("un parlante para escuchar música en el parque", skus_de("altavoces")),
    ("una carpa impermeable para acampar el fin de semana", skus_de("carpas")),
    ("cámara para fotografía con controles manuales", skus_de("camaras")),
    ("silla cómoda para trabajar muchas horas frente al computador", skus_de("sillas")),
]

MODOS = {"denso": buscar_denso, "bm25": buscar_bm25, "hibrido": buscar_hibrido}

def evaluar(modo, k):
    buscar = MODOS[modo]
    hits, rr = [], []
    for consulta, prefijos in EVALSET:
        refs = [r["ref"] for r in buscar(consulta, top_k=k)]
        rank = rank_primer_relevante(refs, prefijos)
        hits.append(1.0 if rank else 0.0)
        rr.append(1.0 / rank if rank else 0.0)
    n = len(EVALSET)
    return {"hit_rate": round(sum(hits) / n, 3), "mrr": round(sum(rr) / n, 3)}

print("Evaluando", len(EVALSET), "consultas × 3 modos × k=1,3,5 …")
filas = []
for modo in MODOS:
    for k in (1, 3, 5):
        filas.append({"modo": modo, "k": k, **evaluar(modo, k)})
tabla = pd.DataFrame(filas).pivot(index="modo", columns="k",
                                  values=["hit_rate", "mrr"]).round(3)
tabla
"""))

    cells.append(md(r"""
### Cómo leer la tabla

- **Columnas**: la misma métrica calculada con distintos cortes $k$. La celda
  (`hibrido`, `hit_rate`, 3) responde: *"con búsqueda híbrida, ¿en qué
  fracción de las consultas hubo algo relevante en el top-3?"*.
- **`hibrido` domina o empata**: combina la comprensión semántica del denso
  con la precisión léxica de BM25. Es el default sensato para producción.
- **`bm25` sufre con las paráfrasis** — "algo para mantener el café caliente"
  no comparte palabras con "termo de acero inoxidable" — y es **ciego a las
  imágenes** (no tienen texto), así que su techo está limitado.
- **HitRate@k sube con $k$** por construcción (un corte más generoso da más
  oportunidades de capturar un relevante). El **MRR casi no cambia con $k$**:
  agrandar el corte agrega resultados *al final*, y el MRR solo depende de la
  posición del primer acierto (solo sube si un relevante que estaba fuera del
  corte entra en él).
- **En qué se diferencian denso e híbrido**: casi solo en la consulta con SKU
  exacto (`TM-1006`) — exactamente el caso que BM25 aporta.
- **Granularidad**: con 16 consultas, cada una mueve las métricas en
  1/16 ≈ 0.06 — diferencias menores a eso son ruido. En producción el evalset
  crece con consultas reales, y se re-ejecuta en CI ante cada cambio de
  chunking, modelo o parámetros: las métricas existen para detectar
  *regresiones*, no como nota final.
- **Límites de estas métricas**: relevancia binaria (no distingue "perfecto"
  de "aceptable") y solo miran el *primer* acierto. Si necesitas medir *cuántos*
  relevantes recuperas o con qué calidad de orden, las siguientes en la caja de
  herramientas son *recall@k* y *nDCG* — mismas ideas, contando más fino.
"""))

    # --- Widgets ---
    cells.append(md(r"""
## 10. Validación interactiva con widgets

Tres paneles con `ipywidgets` para *tocar* el sistema:

1. **Búsqueda por texto** — consultas libres, cambiando modo (denso / BM25 /
   híbrido), `top_k` y filtro de modalidad.
2. **Búsqueda por imagen** — la consulta es una **foto** (una del catálogo o
   una que subas tú): se embebe con el mismo modelo y se busca por similitud
   en el mismo espacio vectorial. Sirve para probar imagen→imagen ("¿cuáles
   se parecen a esta?") e imagen→texto (filtra a "solo texto de productos" y
   verás que una foto encuentra su propia descripción).
3. **Panel de métricas** — recalcula hit rate y MRR del evalset al mover el
   modo y $k$, con el detalle por consulta: en qué posición quedó el primer
   relevante.

> Si los widgets no se muestran, habilita la extensión: en JupyterLab moderno
> `ipywidgets` funciona sin pasos extra; en VS Code usa el renderer de
> notebooks incluido.
"""))

    cells.append(code(r"""
import ipywidgets as W

_q = W.Text(value="zapatillas rojas de lona", description="Consulta:",
            layout=W.Layout(width="450px"))
_modo = W.Dropdown(options=["hibrido", "denso", "bm25"], description="Modo:")
_k = W.IntSlider(value=4, min=1, max=10, description="top_k:")
_tipo = W.Dropdown(options=[("todos", None), ("solo políticas", "doc"),
                            ("solo texto de productos", "producto_texto"),
                            ("solo imágenes", "producto_imagen")],
                   description="Filtro:")
_boton = W.Button(description="Buscar", button_style="primary")
_salida = W.Output()

def _buscar_click(_):
    with _salida:
        _salida.clear_output()
        if _tipo.value is not None and _modo.value != "denso":
            print("(el filtro por tipo usa el índice de Qdrant → aplica en modo denso)")
            resultados = buscar_denso(_q.value, top_k=_k.value, tipo=_tipo.value)
        else:
            resultados = MODOS[_modo.value](_q.value, top_k=_k.value)
        if not resultados:
            print("Sin resultados.")
        else:
            display(mostrar(resultados))

_boton.on_click(_buscar_click)
display(W.VBox([W.HBox([_q, _boton]), W.HBox([_modo, _k, _tipo]), _salida]))
_buscar_click(None)  # primera búsqueda de ejemplo
"""))

    cells.append(code(r"""
# --- Panel 2: búsqueda por IMAGEN (similitud visual) ---
# La consulta ya no es texto: es el embedding de una foto. Nota: si la foto
# elegida está en el índice, ella misma saldrá de primera con score ≈ 1.0 —
# un buen sanity check de que el espacio funciona.
_foto_dd = W.Dropdown(
    options=[(f'{p["sku"]} — {p["nombre"]}', p["imagen"]) for p in PRODUCTS],
    description="Foto:", layout=W.Layout(width="380px"))
_subida = W.FileUpload(accept="image/*", multiple=False, description="…o sube una")
_k_img = W.IntSlider(value=5, min=1, max=10, description="top_k:")
_tipo_img = W.Dropdown(options=[("todos", None), ("solo imágenes", "producto_imagen"),
                                ("solo texto de productos", "producto_texto"),
                                ("solo políticas", "doc")],
                       description="Filtro:")
_btn_img = W.Button(description="Buscar parecidos", button_style="primary")
_out_img = W.Output()

def _imagen_de_consulta():
    # Prioridad: archivo subido; si no hay, la foto elegida del catálogo.
    if _subida.value:
        item = (_subida.value[0] if isinstance(_subida.value, (list, tuple))
                else list(_subida.value.values())[0])         # ipywidgets 8 / 7
        data = bytes(item["content"])
        mime = item.get("type") or "image/jpeg"
        return data, mime, f"imagen subida ({item.get('name', '?')})"
    ruta = CATALOG_DIR / "images" / _foto_dd.value
    return ruta.read_bytes(), "image/jpeg", _foto_dd.value

def _buscar_imagen_click(_):
    with _out_img:
        _out_img.clear_output()
        data, mime, etiqueta = _imagen_de_consulta()
        display(HTML(
            f'<div style="font-family:sans-serif;font-size:12px">Consulta: '
            f'<b>{etiqueta}</b><br><img src="data:{mime};base64,'
            f'{base64.b64encode(data).decode()}" '
            f'style="width:110px;border-radius:8px;margin:4px 0"></div>'))
        vec = _embed([types.Part.from_bytes(data=data, mime_type=mime)])
        display(mostrar(buscar_por_vector(vec, top_k=_k_img.value, tipo=_tipo_img.value)))

_btn_img.on_click(_buscar_imagen_click)
display(W.VBox([W.HBox([_foto_dd, _subida]),
                W.HBox([_k_img, _tipo_img, _btn_img]), _out_img]))
_buscar_imagen_click(None)  # ejemplo inicial con la primera foto del catálogo
"""))

    cells.append(code(r"""
# --- Panel 3: métricas en vivo — mueve modo y k, y mira el detalle por consulta.
_modo_m = W.Dropdown(options=["hibrido", "denso", "bm25"], description="Modo:")
_k_m = W.IntSlider(value=3, min=1, max=10, description="k:")
_salida_m = W.Output()

def _refrescar_metricas(*_):
    with _salida_m:
        _salida_m.clear_output()
        buscar = MODOS[_modo_m.value]
        detalle = []
        for consulta, prefijos in EVALSET:
            refs = [r["ref"] for r in buscar(consulta, top_k=_k_m.value)]
            rank = rank_primer_relevante(refs, prefijos)
            etiqueta = "|".join(prefijos[:3]) + ("…" if len(prefijos) > 3 else "")
            detalle.append({
                "consulta": consulta[:48] + ("…" if len(consulta) > 48 else ""),
                "relevantes": etiqueta,
                "rank_1er_relevante": rank if rank else "—",
                "recíproco": round(1.0 / rank, 3) if rank else 0.0,
            })
        df = pd.DataFrame(detalle)
        agg = evaluar(_modo_m.value, _k_m.value)
        print(f"modo={_modo_m.value}  k={_k_m.value}  →  "
              f"HitRate@{_k_m.value}={agg['hit_rate']}   MRR@{_k_m.value}={agg['mrr']}\n")
        display(df)

_modo_m.observe(_refrescar_metricas, names="value")
_k_m.observe(_refrescar_metricas, names="value")
display(W.VBox([W.HBox([_modo_m, _k_m]), _salida_m]))
_refrescar_metricas()
"""))

    cells.append(md(r"""
## Resumen

- `gemini-embedding-2` proyecta **texto e imágenes al mismo espacio**: la
  consulta escrita recupera fotos directamente (RAG multimodal sin
  anotaciones intermedias). Ojo con sus convenciones: formato asimétrico
  consulta/documento, y la **agregación** cuando `contents` es una lista.
- **Qdrant** guarda vectores + payloads; el campo `tipo` del payload nos dio
  filtrado por modalidad, y los IDs deterministas ingesta idempotente.
- El **chunking** es una decisión de diseño (tamaño, solapamiento, dónde
  cortar); `RecursiveCharacterTextSplitter` respeta las fronteras naturales.
- La búsqueda **híbrida** (densa + BM25 vía RRF) une paráfrasis y términos
  exactos; las imágenes solo son alcanzables por la rama densa.
- La generación es una cadena LCEL: recuperar → prompt con citas → `ChatOllama`
  (el modelo de Ollama cloud se elige en el notebook, con fallback automático).
- **Hit rate** (¿apareció algo relevante?) y **MRR** (¿qué tan arriba?) miden
  la recuperación sin tocar el LLM; los widgets permiten validar a mano lo que
  las métricas dicen en agregado.

**Siguiente** → `02_agentes_langgraph.ipynb`: convertimos este pipeline en una
*herramienta* que un agente LangGraph decide cuándo y cómo usar.
"""))

    return cells


# ===========================================================================
# Notebook 2 — Agentes con LangGraph (el RAG como herramienta)
# ===========================================================================
def build_notebook_2() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# 02 — Agentes con LangGraph: el RAG como herramienta

Segundo notebook de la serie incremental. En el 01 construimos un pipeline de
RAG multimodal sobre Qdrant; aquí lo convertimos en una **herramienta** que un
**agente** decide cuándo usar.

> **Prerrequisito:** haber ejecutado el notebook 01 (la colección
> `tecnomarket` debe existir en Qdrant) y tener el mismo `.env`.

## ¿Qué es un agente?

Una llamada simple a un LLM es de un solo turno: entra un prompt, sale texto.
Un **agente** envuelve al modelo en un loop **ReAct** (*Reason + Act*, el
patrón del paper [ReAct, 2022](https://arxiv.org/abs/2210.03629)):

```
 ┌─────────────────────────────────────────────────┐
 │  reasoning → action (tool call) → observation   │
 │      ▲                                 │        │
 │      └───────────── repetir ───────────┘        │
 └──────────── hasta que el modelo responde ───────┘
```

Los ingredientes: **tools** (funciones que el modelo puede invocar vía *tool
calling*), **estado** (la conversación acumulada) y una **condición de
parada** (el modelo responde sin pedir tools).

## RAG fijo vs. RAG agéntico

En el notebook 01 el flujo era **fijo**: *siempre* recuperamos top-k y
generamos — aunque la pregunta no lo necesite, y exactamente una vez.

Con el RAG **como tool**, el modelo **decide**:

- *si* buscar (una pregunta de aritmética no necesita el índice),
- *qué* buscar (reformula la consulta en sus propios términos),
- *cuántas veces* buscar (preguntas compuestas → varias búsquedas),
- y puede **combinar** tools (buscar la política de envíos *y* calcular
  un total).

Ese es el salto de "pipeline" a "agente", y LangGraph lo modela como una
**máquina de estados explícita**: **nodes** (unidades de trabajo) y **edges**
(transiciones), con edges *condicionales* para el routing — lo vemos pieza
por pieza en la sección 2.

## Lo que cubrimos

1. El núcleo RAG del notebook 01, compactado.
2. **LangGraph 101** — qué es un node, un edge, el estado, y cómo se
   construye y extiende un `StateGraph`.
3. **Tools** — el RAG como tool, más tools creativas (tracking de pedidos,
   filtro por presupuesto, calculadora).
4. El **grafo del agente** y su loop ReAct.
5. **Playground interactivo** — ver los rounds del agente y el StateGraph
   iluminarse al mismo tiempo.
6. **Reasoning** — activar el *thinking* del modelo y leer su razonamiento.
7. **Fuentes multimodales** — las fotos de producto que el agente recuperó.
8. **Memoria** — checkpointer + `thread_id` para conversaciones multi-turno.
9. **Streamlit app** — todo junto en una app con Docker (`app/`).
"""))

    cells.append(code(CONFIG_CELL))

    cells.append(md(r"""
## 1. El núcleo de RAG, en versión compacta

Re-empacamos la lógica del notebook 01 en unas pocas funciones: embeddings de
consulta (Gemini), búsqueda densa (Qdrant), BM25 (reconstruido leyendo los
payloads de la colección) y fusión RRF. Nada nuevo — solo más denso, porque
ahora el protagonista es el agente.

También cargamos el **catálogo con sus fotos** (como en el notebook 01):
las usaremos para que el agente muestre *fuentes visuales* (sección 7) y
para las tools estructuradas (sección 3).
"""))

    cells.append(code(r"""
import time
from functools import lru_cache

from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

gclient = genai.Client()
qdrant = QdrantClient(url=QDRANT_URL)

if not qdrant.collection_exists(COLLECTION):
    raise RuntimeError(
        f"La colección '{COLLECTION}' no existe. Ejecuta primero el notebook "
        "01_rag_multimodal.ipynb (sección de ingesta)."
    )

@lru_cache(maxsize=512)
def embed_consulta(texto):
    for intento in range(5):
        try:
            r = gclient.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=f"task: search result | query: {texto}",
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
            )
            return list(r.embeddings[0].values)
        except Exception:
            time.sleep(2 ** intento)
    raise RuntimeError("API de embeddings sin respuesta.")

# Reconstruimos el corpus léxico desde los payloads de Qdrant (en el notebook
# 01 lo teníamos en memoria; aquí la colección es la fuente de la verdad).
_puntos, _ = qdrant.scroll(collection_name=COLLECTION, limit=1000, with_payload=True)
PAYLOAD_POR_REF = {p.payload["ref"]: p.payload for p in _puntos}
_corpus = [(ref, pl["texto"]) for ref, pl in PAYLOAD_POR_REF.items() if pl["texto"]]
_refs_bm25 = [ref for ref, _ in _corpus]
_bm25 = BM25Okapi([t.lower().split() for _, t in _corpus])

def buscar_hibrido(consulta, top_k=4, pool=15, rrf_k=60):
    densos = qdrant.query_points(collection_name=COLLECTION,
                                 query=embed_consulta(consulta),
                                 limit=pool, with_payload=True).points
    scores = _bm25.get_scores(consulta.lower().split())
    orden_lex = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    fusion = {}
    for rank, h in enumerate(densos):
        fusion[h.payload["ref"]] = fusion.get(h.payload["ref"], 0) + 1 / (rrf_k + rank)
    for rank, i in enumerate(orden_lex[:pool]):
        if scores[i] > 0:
            ref = _refs_bm25[i]
            fusion[ref] = fusion.get(ref, 0) + 1 / (rrf_k + rank)
    mejores = sorted(fusion, key=fusion.get, reverse=True)[:top_k]
    return [PAYLOAD_POR_REF[ref] for ref in mejores]

print("Núcleo RAG listo. Puntos en la colección:", len(PAYLOAD_POR_REF))
"""))

    cells.append(code(r"""
# Catálogo + helpers de imágenes (mismos trucos del notebook 01).
import base64
import json

CATALOG_DIR = ROOT / "rag" / "catalog"
PRODUCTS = json.loads((CATALOG_DIR / "products.json").read_text(encoding="utf-8"))
PROD_POR_SKU = {p["sku"]: p for p in PRODUCTS}

from IPython.display import HTML, display

def img_b64(path):
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()

def galeria(productos, ancho=120):
    tarjetas = []
    for p in productos:
        src = img_b64(CATALOG_DIR / "images" / p["imagen"])
        tarjetas.append(
            f'<div style="display:inline-block;margin:6px;text-align:center;width:{ancho}px;'
            f'vertical-align:top;font-family:sans-serif;font-size:11px">'
            f'<img src="{src}" style="width:{ancho}px;height:{ancho}px;object-fit:cover;'
            f'border-radius:8px"><br><b>{p["sku"]}</b><br>{p["nombre"]}<br>'
            f'${p["precio"]:,.0f}</div>'
        )
    return HTML("<div>" + "".join(tarjetas) + "</div>")

print(f"{len(PRODUCTS)} productos cargados con foto y precio.")
display(galeria(PRODUCTS[:4]))
"""))

    cells.append(md(r"""
## 2. LangGraph 101: estado, nodes y edges

LangGraph modela un agente como un **grafo de estados** (`StateGraph`). Antes
de meter un LLM, entendamos las tres piezas con un ejemplo de juguete:

### El estado (State)

El estado es **un diccionario compartido** que viaja por el grafo: cada node
lo lee y devuelve los campos que quiere cambiar. Se declara como `TypedDict`
(un dict con campos y tipos fijos).

### ¿Qué es un reducer?

Cuando un node devuelve un campo que **ya tenía valor**, hay que decidir qué
hacer con los dos valores (el viejo y el nuevo). Esa decisión es el
**reducer**: una función `(viejo, nuevo) → resultado`.

Solo hay dos casos que nos importan:

| Campo declarado como… | El node devuelve `x` | Resultado |
|---|---|---|
| `x: str` (sin reducer) | `"hola"` | `"hola"` — **reemplaza** lo que había |
| `x: Annotated[list, operator.add]` | `["hola"]` | `viejo + ["hola"]` — **se suma a la lista** |

Es decir: sin reducer, el último node "gana"; con reducer, los aportes de
todos los nodes **se acumulan**. Para un chat queremos lo segundo — que cada
mensaje se agregue al historial, no que lo borre — y para eso existe
`add_messages`, el reducer de LangGraph para listas de mensajes
(esencialmente `viejo + nuevo`, con detalles extra como deduplicar por id).

La sintaxis `Annotated[list, operator.add]` se lee: *"este campo es una
`list`, y su reducer es `operator.add`"* — `Annotated` solo adjunta ese dato
extra al tipo.

### El node

Una **función de Python** que recibe el estado y devuelve un **update
parcial** (un dict con solo los campos que cambia). Eso es todo — un node no
sabe nada del resto del grafo. Se registra con `add_node("nombre", funcion)`.

### El edge

La **transición** entre nodes. Dos tipos:

- **Edge fijo** — `add_edge("a", "b")`: después de `a` SIEMPRE viene `b`.
- **Edge condicional** — `add_conditional_edges("a", router, mapa)`: después
  de `a` se ejecuta la función `router(estado)`, que devuelve una etiqueta, y
  el `mapa` traduce esa etiqueta al node destino. Así se implementa el
  *branching* (y el loop del agente).

Dos nodes especiales: `START` (por dónde entra el input) y `END` (terminar).

### La receta para construir (o extender) un grafo

```python
grafo = StateGraph(MiEstado)        # 1. declarar el estado
grafo.add_node("a", nodo_a)         # 2. registrar nodes
grafo.add_node("b", nodo_b)         #    (extender = añadir más add_node
grafo.add_edge(START, "a")          #     y cablearlos con edges ANTES
grafo.add_conditional_edges(...)    #     de compilar)
grafo.add_edge("b", END)
app = grafo.compile()               # 3. compilar → objeto ejecutable
```

⚠ **`compile()` congela el grafo**: el objeto compilado es inmutable. Para
extenderlo (otro node, otra tool) se agrega al *builder* y se recompila —
por eso conviene encapsular la construcción en una función.
"""))

    cells.append(code(r"""
# Grafo de juguete SIN LLM, para ver la mecánica pura.
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

class EstadoDemo(TypedDict):
    texto: str                              # sin reducer → se sobrescribe
    trazas: Annotated[list[str], operator.add]  # con reducer → se acumula

# --- Nodes: funciones estado → update parcial ---
def normalizar(estado: EstadoDemo) -> dict:
    return {"texto": estado["texto"].strip().lower(),
            "trazas": ["normalizar: limpié el texto"]}

def responder_pregunta(estado: EstadoDemo) -> dict:
    return {"texto": f"Buena pregunta: '{estado['texto']}'",
            "trazas": ["responder_pregunta"]}

def hacer_eco(estado: EstadoDemo) -> dict:
    return {"texto": estado["texto"].upper() + "!!!",
            "trazas": ["hacer_eco"]}

# --- Router del edge condicional: devuelve una ETIQUETA, no un node ---
def es_pregunta(estado: EstadoDemo) -> str:
    return "pregunta" if estado["texto"].endswith("?") else "afirmacion"

# --- Construcción ---
demo = StateGraph(EstadoDemo)
demo.add_node("normalizar", normalizar)
demo.add_node("responder_pregunta", responder_pregunta)
demo.add_node("hacer_eco", hacer_eco)
demo.add_edge(START, "normalizar")
demo.add_conditional_edges("normalizar", es_pregunta,
                           {"pregunta": "responder_pregunta",
                            "afirmacion": "hacer_eco"})
demo.add_edge("responder_pregunta", END)
demo.add_edge("hacer_eco", END)
app_demo = demo.compile()

for entrada in ["  ¿Hay envío gratis?  ", "me gusta esta tienda"]:
    final = app_demo.invoke({"texto": entrada, "trazas": []})
    print(f"{entrada!r:32} → {final['texto']!r}")
    print(f"{'':32}   trazas acumuladas: {final['trazas']}")
"""))

    cells.append(code(r"""
# El grafo se dibuja solo (Mermaid). Compara el diagrama con el código:
# cajas = nodes, flechas sólidas = edges fijos, punteadas = condicionales.
from IPython.display import Image, display

try:
    display(Image(app_demo.get_graph().draw_mermaid_png()))
except Exception:
    print(app_demo.get_graph().draw_ascii())
"""))

    cells.append(code(r"""
# EXTENDER un grafo = volver al builder, añadir el node, recablear, recompilar.
# Insertamos un node "censurar" entre los dos finales y END.
#
# ⚠ LangGraph avisará "Adding a node to a graph that has already been
# compiled…" — exactamente el punto: app_demo (ya compilado) NO cambia;
# los cambios solo existen en el builder hasta el nuevo compile().
def censurar(estado: EstadoDemo) -> dict:
    return {"texto": estado["texto"].replace("!!!", "."),
            "trazas": ["censurar: bajé el tono"]}

demo.add_node("censurar", censurar)
demo.edges.discard(("responder_pregunta", END))   # quitamos los edges viejos
demo.edges.discard(("hacer_eco", END))
demo.add_edge("responder_pregunta", "censurar")
demo.add_edge("hacer_eco", "censurar")
demo.add_edge("censurar", END)
app_demo_v2 = demo.compile()                      # nueva versión ejecutable

print(app_demo_v2.invoke({"texto": "me gusta esta tienda", "trazas": []})["texto"])
try:
    display(Image(app_demo_v2.get_graph().draw_mermaid_png()))
except Exception:
    print(app_demo_v2.get_graph().draw_ascii())
"""))

    cells.append(md(r"""
Con esa mecánica clara, el agente es solo un caso particular: un node que
llama al LLM, un node que ejecuta tools, y un edge condicional que pregunta
*"¿el modelo pidió tools?"* para cerrar el loop ReAct.

## 3. Las tools

Cinco tools declaradas con `@tool` de LangChain. El **docstring y los type
hints se convierten en el JSON Schema** que el modelo ve — descríbelos
pensando en el modelo, no en humanos: de esa descripción depende que el agente
sepa *cuándo* invocar cada una.

| Tool | Qué enseña |
|---|---|
| `buscar_tecnomarket` | el RAG del notebook 01, ahora como tool (búsqueda semántica) |
| `detalle_producto` | lookup exacto por SKU (filtro de payload, sin embeddings) |
| `productos_por_presupuesto` | filtro **estructurado** (categoría + precio): no todo es semántico |
| `estado_pedido` | envolver un "backend" (aquí simulado): tracking de pedidos |
| `calculadora` | combinar conocimiento recuperado con cálculo |
"""))

    cells.append(code(r"""
import hashlib
from datetime import date, timedelta

from langchain_core.tools import tool

# Registro de fuentes: cada búsqueda anota los SKUs que recuperó, para poder
# mostrar las FOTOS de esos productos después de la respuesta (sección 7).
FUENTES_RECIENTES: list[dict] = []

@tool
def buscar_tecnomarket(consulta: str) -> str:
    'Busca en la base de conocimiento de TecnoMarket (políticas de envíos, devoluciones y garantías, y catálogo de productos). Recibe una consulta en lenguaje natural y devuelve los pasajes más relevantes.'
    resultados = buscar_hibrido(consulta, top_k=4)
    FUENTES_RECIENTES.extend(r for r in resultados if r.get("sku"))
    bloques = []
    for i, r in enumerate(resultados, 1):
        if r["tipo"] == "producto_imagen":
            bloques.append(f"[{i}] foto del producto {r['sku']} — {r['nombre']}")
        else:
            origen = r.get("fuente") or f"{r['sku']} — {r['nombre']}"
            bloques.append(f"[{i}] (fuente: {origen})\n{r['texto']}")
    return "\n\n".join(bloques) if bloques else "Sin resultados."

@tool
def detalle_producto(sku: str) -> str:
    'Devuelve la ficha exacta de un producto de TecnoMarket dado su SKU (formato TM-XXXX): nombre, categoría, precio y descripción.'
    p = PROD_POR_SKU.get(sku.strip().upper())
    if not p:
        return f"No existe el SKU {sku}."
    FUENTES_RECIENTES.append({"sku": p["sku"], "nombre": p["nombre"],
                              "imagen": p["imagen"]})
    return json.dumps({k: p[k] for k in ("sku", "nombre", "categoria",
                                         "precio", "descripcion")},
                      ensure_ascii=False)

@tool
def productos_por_presupuesto(categoria: str, presupuesto: float) -> str:
    'Lista los productos de TecnoMarket de una categoría (p. ej. "audifonos", "zapatillas", "relojes", "cafeteras") con precio menor o igual al presupuesto dado, ordenados del más barato al más caro.'
    cat = categoria.strip().lower()
    encontrados = sorted(
        (p for p in PRODUCTS
         if p["categoria"] == cat and p["precio"] <= presupuesto),
        key=lambda p: p["precio"])[:5]
    if not encontrados:
        categorias = sorted({p["categoria"] for p in PRODUCTS})
        return (f"Nada en '{categoria}' por ≤ {presupuesto}. "
                f"Categorías disponibles: {', '.join(categorias)}.")
    FUENTES_RECIENTES.extend({"sku": p["sku"], "nombre": p["nombre"],
                              "imagen": p["imagen"]} for p in encontrados)
    return json.dumps([{k: p[k] for k in ("sku", "nombre", "precio")}
                       for p in encontrados], ensure_ascii=False)

@tool
def estado_pedido(numero_pedido: str) -> str:
    'Consulta el estado de un pedido de TecnoMarket dado su número (formato PED-XXXX): estado actual, transportadora y fecha estimada de entrega.'
    # Simulación determinista: el hash del número decide el estado. En
    # producción esta tool llamaría al API interno de órdenes — al agente
    # le da igual: solo ve el docstring y el resultado.
    n = int(hashlib.sha1(numero_pedido.strip().upper().encode()).hexdigest(), 16)
    estados = ["confirmado, preparando el paquete", "despachado, en tránsito",
               "en el centro de distribución local", "en reparto, llega hoy"]
    entrega = date.today() + timedelta(days=n % 4 + 1)
    return json.dumps({"pedido": numero_pedido.strip().upper(),
                       "estado": estados[n % len(estados)],
                       "transportadora": ["VelozExpress", "AndesCargo"][n % 2],
                       "entrega_estimada": entrega.isoformat()},
                      ensure_ascii=False)

@tool
def calculadora(expresion: str) -> str:
    'Evalúa una expresión aritmética simple, p. ej. "2 * 129.0 + 12000".'
    if not set(expresion) <= set("0123456789+-*/(). "):
        return "Error: solo se admite aritmética básica."
    try:
        return str(eval(expresion, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error: {exc}"

TOOLS = [buscar_tecnomarket, detalle_producto, productos_por_presupuesto,
         estado_pedido, calculadora]
print("Tools:", [t.name for t in TOOLS])
print("\nPruebas directas (sin LLM):")
print(" ", detalle_producto.invoke({"sku": "TM-1008"})[:100], "…")
print(" ", productos_por_presupuesto.invoke(
    {"categoria": "audifonos", "presupuesto": 200}))
print(" ", estado_pedido.invoke({"numero_pedido": "PED-7431"}))
"""))

    cells.append(md(r"""
## 4. El LLM (Ollama cloud) con las tools enlazadas

Misma celda de conexión del notebook 01 (elige el modelo en `OLLAMA_MODEL`;
para agentes conviene uno con buen *tool calling*). `bind_tools` adjunta los
JSON Schemas de las tools a cada llamada, habilitando el *tool calling*
nativo del modelo.
"""))

    cells.append(code(LLM_CELL))

    cells.append(md(r"""
## 5. El grafo del agente

Con el vocabulario de la sección 2, el agente es un `StateGraph` de **dos
nodes y un loop**:

- **Estado**: la lista `messages`, con el reducer `add_messages` (cada node
  *agrega* mensajes en lugar de sobrescribir la lista).
- **Node `agente`**: llama al LLM (con las tools enlazadas), anteponiendo el
  **system prompt** — en un agente no hay system prompt implícito: el modelo
  ve exactamente los mensajes que le pasamos, así que lo inyectamos nosotros
  en cada llamada.
- **Node `tools`**: `ToolNode` (prefabricado de LangGraph) ejecuta las tools
  que el modelo pidió y agrega sus resultados como `ToolMessage`s.
- **Edge condicional**: `tools_condition` rutea — ¿el último mensaje pide
  tools? → `tools`; ¿no? → `END`.
- **Edge fijo `tools → agente`**: cierra el loop ReAct.

```
START → agente ─(¿pidió tools?)─ sí → tools ─┐
          ▲               │                  │
          └───────────────┼──────────────────┘
                          no → END
```
"""))

    cells.append(code(r"""
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class EstadoAgente(TypedDict):
    messages: Annotated[list, add_messages]

# ¿Y el system prompt? En un agente NO aparece solo: el modelo recibe
# exactamente estado["messages"]. Lo definimos aquí y el node agente lo
# antepone EN CADA LLAMADA (sin guardarlo en el estado: así no se duplica
# en la memoria y se puede cambiar sin tocar los checkpoints).
# Ojo: las descripciones de las tools NO van aquí — bind_tools las envía
# aparte, en el campo `tools` del API (inspecciónalas con
# llm.bind_tools(TOOLS).kwargs["tools"]).
SYSTEM_PROMPT = (
    "Eres el asistente de la tienda TecnoMarket. Responde en español, breve "
    "y amable. Usa las tools para consultar catálogo, políticas y pedidos: "
    "no inventes SKUs, precios ni plazos. Si no encuentras la información, "
    "dilo claramente.")

def construir_agente(llm_base, checkpointer=None):
    # Encapsular la construcción (sección 2) permite recompilar variantes:
    # otro LLM (reasoning, sección 6) u otra memoria (sección 8).
    llm_con_tools = llm_base.bind_tools(TOOLS)

    def nodo_agente(estado: EstadoAgente) -> dict:
        mensajes = [SystemMessage(content=SYSTEM_PROMPT)] + estado["messages"]
        return {"messages": [llm_con_tools.invoke(mensajes)]}

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("tools", ToolNode(TOOLS))
    grafo.add_edge(START, "agente")
    grafo.add_conditional_edges("agente", tools_condition,
                                {"tools": "tools", END: END})
    grafo.add_edge("tools", "agente")
    return grafo.compile(checkpointer=checkpointer)

agente = construir_agente(llm)
print("Grafo compilado.")

try:
    display(Image(agente.get_graph().draw_mermaid_png()))
except Exception:
    print(agente.get_graph().draw_ascii())
"""))

    cells.append(md(r"""
## 6. Playground: los rounds del agente y el StateGraph, en vivo

Cada iteración del loop ReAct es un **round**: se ejecuta un node y el estado
crece. Con `stream_mode="updates"` el grafo emite `{node: update}` tras cada
node — sabemos exactamente **dónde está** la ejecución.

El panel de abajo junta las dos vistas:

- **Izquierda** — el StateGraph con el node activo **iluminado**.
- **Derecha** — el transcript: tool calls (con argumentos), resultados de las
  tools y la respuesta final.

Y se puede avanzar **round a round** (botón *⏭ Paso*) — ideal para narrar en
clase qué decidió el modelo en cada transición — o dejarlo correr (*▶ Hasta
el final*).

Preguntas sugeridas (cada una ejercita tools distintas):

- *¿Cuánto costarían dos teclados TM-1008 con el envío estándar incluido?* →
  ficha + búsqueda + calculadora.
- *¿Puedo devolver unos audífonos abiertos? ¿y cuántos meses de garantía
  tienen?* → dos búsquedas reformuladas (RAG agéntico).
- *¿Dónde está mi pedido PED-7431?* → tracking simulado.
- *Recomiéndame audífonos por menos de 200* → filtro estructurado.
- *¿Cuánto es 15% de 549?* → sin tools de búsqueda: el índice no hace falta.
"""))

    cells.append(code(r"""
import html as html_mod

import ipywidgets as W

NODOS_VISTA = ["START", "agente", "tools", "END"]

def render_grafo(activo=None):
    # Dibujo HTML del StateGraph con el node activo resaltado. (Para grafos
    # grandes usaríamos draw_mermaid_png; aquí lo pintamos a mano para poder
    # iluminarlo en cada round.)
    def caja(n):
        on = n == activo
        estilo = ("background:#ff6f00;color:white;border:2px solid #e65100"
                  if on else "background:#f5f5f5;color:#333;border:1px solid #bbb")
        forma = "border-radius:16px" if n in ("START", "END") else "border-radius:6px"
        return (f'<div style="{estilo};{forma};padding:6px 18px;margin:4px auto;'
                f'width:90px;text-align:center;font-family:monospace;'
                f'font-size:13px">{n}</div>')
    flecha = '<div style="text-align:center;color:#888">↓</div>'
    loop = ('<div style="text-align:center;color:#888;font-size:11px">'
            '⇄ (¿pidió tools? sí → tools → agente; no → END)</div>')
    return (f'<div style="width:270px">{caja("START")}{flecha}{caja("agente")}'
            f'{loop}{caja("tools")}{flecha}{caja("END")}</div>')

def describir(msg):
    # Un mensaje del estado → una línea legible del transcript.
    if isinstance(msg, AIMessage):
        piezas = []
        razonamiento = msg.additional_kwargs.get("reasoning_content")
        if razonamiento:
            piezas.append(f"🧠 <i>{html_mod.escape(razonamiento[:300])}…</i>")
        for tc in msg.tool_calls:
            piezas.append(f"🛠 <b>tool call</b> → <code>{tc['name']}"
                          f"({json.dumps(tc['args'], ensure_ascii=False)})</code>")
        if msg.content:
            piezas.append(f"💬 <b>respuesta:</b> {html_mod.escape(msg.content)}")
        return "<br>".join(piezas)
    if isinstance(msg, ToolMessage):
        return (f"📦 <code>{msg.name}</code> devolvió: "
                f"<span style='color:#666'>{html_mod.escape(str(msg.content)[:280])}…</span>")
    return html_mod.escape(str(msg.content))

_pg_q = W.Text(value="¿Cuánto costarían dos teclados TM-1008 con el envío estándar incluido?",
               layout=W.Layout(width="640px"), description="Pregunta:")
_pg_nueva = W.Button(description="🔄 Nueva pregunta", button_style="primary")
_pg_paso = W.Button(description="⏭ Paso", disabled=True)
_pg_todo = W.Button(description="▶ Hasta el final", disabled=True)
_pg_grafo = W.HTML(value=render_grafo())
_pg_log = W.Output(layout=W.Layout(border="1px solid #ddd", width="640px",
                                   height="380px", overflow="auto"))
_pg_iter = None
_pg_round = 0

def _log_html(texto):
    with _pg_log:
        display(HTML(f'<div style="font-family:sans-serif;font-size:12px;'
                     f'margin:4px 6px">{texto}</div>'))

def _pg_reset(_):
    global _pg_iter, _pg_round
    _pg_log.clear_output()
    _pg_round = 0
    _pg_iter = agente.stream(
        {"messages": [HumanMessage(content=_pg_q.value)]}, stream_mode="updates")
    _pg_grafo.value = render_grafo("START")
    _log_html(f"👤 <b>{html_mod.escape(_pg_q.value)}</b>")
    _pg_paso.disabled = _pg_todo.disabled = False

def _pg_avanzar(_):
    global _pg_iter, _pg_round
    item = next(_pg_iter, None)
    if item is None:
        _pg_grafo.value = render_grafo("END")
        _log_html("🏁 <b>END</b> — el modelo respondió sin pedir más tools.")
        _pg_paso.disabled = _pg_todo.disabled = True
        return False
    nodo, update = next(iter(item.items()))
    _pg_round += 1
    _pg_grafo.value = render_grafo(nodo)
    _log_html(f'<span style="color:#ff6f00"><b>— round {_pg_round}: '
              f'node <code>{nodo}</code> —</b></span>')
    for msg in update["messages"]:
        _log_html(describir(msg))
    return True

def _pg_correr(_):
    while _pg_avanzar(None):
        pass

_pg_nueva.on_click(_pg_reset)
_pg_paso.on_click(_pg_avanzar)
_pg_todo.on_click(_pg_correr)
display(W.VBox([_pg_q, W.HBox([_pg_nueva, _pg_paso, _pg_todo]),
                W.HBox([_pg_grafo, _pg_log])]))
_pg_reset(None)  # deja la primera pregunta lista: pulsa ⏭ Paso round a round
"""))

    cells.append(md(r"""
Fíjate en el segundo ejemplo (la pregunta compuesta de devoluciones +
garantías): el agente suele lanzar **dos tool calls con consultas
reformuladas** en lugar de una sola mezclada — eso es **RAG agéntico**, lo
que el pipeline fijo del notebook 01 no podía hacer.

## 7. Reasoning: ver al modelo *pensar*

`minimax-m3` es un modelo **razonador** (*thinking model*): antes de la
respuesta genera *tokens de razonamiento* — un borrador interno donde planea,
se corrige y decide. Por defecto `ChatOllama` no lo pide; se activa con
`reasoning=True`, y el texto llega **separado** de la respuesta, en
`additional_kwargs["reasoning_content"]`.

¿Por qué importa en agentes? El razonamiento muestra **por qué eligió una
tool** ("el usuario pregunta por un precio → necesito la ficha…"), que es
justo lo que queremos auditar (y enseñar). El costo: más tokens y más
latencia — en producción se activa selectivamente.

> Si elegiste un modelo sin thinking (p. ej. `gpt-oss` responde razonamiento
> vacío o no soporta el flag), la celda lo detecta y lo dice.
"""))

    cells.append(code(r"""
# El mismo conectar_llm de siempre, ahora con reasoning=True.
llm_razonador = conectar_llm(reasoning=True)

resp = llm_razonador.invoke(
    "Tengo 3 pedidos de 2 teclados cada uno a 129 USD el teclado, con 10% de "
    "descuento sobre el total. ¿Cuánto pago? Responde en una frase.")

razonamiento = resp.additional_kwargs.get("reasoning_content", "")
if razonamiento:
    print("=== REASONING (borrador interno del modelo) ===")
    print(razonamiento[:1200])
    print("\n=== RESPUESTA (lo que vería el usuario) ===")
else:
    print(f"(El modelo '{llm_razonador.model}' no expone reasoning; "
          "prueba con minimax-m3:cloud.)\n")
print(resp.content)
"""))

    cells.append(code(r"""
# Reasoning DENTRO del agente: recompilamos el grafo con el LLM razonador
# (por eso construir_agente recibe el LLM como parámetro). El playground de
# la sección 6 ya muestra el 🧠 cuando un AIMessage trae reasoning_content:
# re-apunta la variable y vuelve a usarlo si quieres verlo con widgets.
agente_razonador = construir_agente(llm_razonador)

for paso in agente_razonador.stream(
        {"messages": [HumanMessage(content=
            "¿Me conviene más el pedido PED-7431 o comprar de nuevo unos "
            "audífonos de menos de 150 USD? Revisa el estado del pedido y "
            "las opciones antes de responder.")]},
        stream_mode="updates"):
    nodo, update = next(iter(paso.items()))
    print(f"\n════ node: {nodo} ════")
    for msg in update["messages"]:
        r = getattr(msg, "additional_kwargs", {}).get("reasoning_content")
        if r:
            print(f"🧠 reasoning: {r[:400]}…\n")
        msg.pretty_print()
"""))

    cells.append(md(r"""
## 8. Fuentes multimodales: mostrar lo que el agente recuperó

Como en el notebook 01, no basta con leer la respuesta: hay que **auditar las
fuentes**. Las tools registran los productos que tocaron
(`FUENTES_RECIENTES`), así que después de cada corrida podemos renderizar la
**galería de fotos** de los productos que respaldan la respuesta — RAG
multimodal, ahora dentro del agente.
"""))

    cells.append(code(r"""
def preguntar_con_fuentes(pregunta, agente_a_usar=None):
    ag = agente_a_usar or agente
    FUENTES_RECIENTES.clear()
    final = ag.invoke({"messages": [HumanMessage(content=pregunta)]})
    print(final["messages"][-1].content)
    vistos, productos = set(), []
    for f in FUENTES_RECIENTES:
        p = PROD_POR_SKU.get(f["sku"])
        if p and p["sku"] not in vistos:
            vistos.add(p["sku"])
            productos.append(p)
    if productos:
        display(HTML("<hr><b>🖼 Productos consultados por el agente:</b>"))
        display(galeria(productos[:6]))

preguntar_con_fuentes(
    "Busco unas zapatillas de lona para uso diario y quiero saber si podría "
    "devolverlas en caso de que no me queden bien.")
"""))

    cells.append(md(r"""
## 9. Memoria: checkpointer + `thread_id`

Hasta ahora cada `invoke` **empieza de cero**: el estado vive solo durante la
corrida. Pregunta "¿cuánto cuesta el TM-1008?" y luego "¿y su garantía?" — el
agente no sabe de qué producto hablas.

LangGraph resuelve esto con un **checkpointer**. Desarmemos la pieza:

### ¿QUÉ se guarda exactamente?

**El estado completo del grafo** — en nuestro caso, la lista `messages`
entera. Y ojo: eso no es solo "pregunta y respuesta"; es **todo el tráfico
del loop ReAct**:

| # | Tipo de mensaje | Contenido |
|---|---|---|
| 1 | `HumanMessage` | "¿Cuánto cuesta el teclado TM-1008?" |
| 2 | `AIMessage` | tool call → `detalle_producto({"sku": "TM-1008"})` |
| 3 | `ToolMessage` | el JSON que devolvió la tool (¡con el precio!) |
| 4 | `AIMessage` | la respuesta final al usuario |

Los 4 mensajes quedan guardados. Por eso el turno 2 puede resolver "¿y su
garantía?": el modelo **re-lee** ese historial (incluido el `ToolMessage` con
la ficha del TM-1008) antes de responder.

### ¿CUÁNDO y DÓNDE se guarda?

- **Cuándo**: después de **cada node** (no al final de la corrida), el
  checkpointer persiste una foto del estado — un **checkpoint**. Una corrida
  con 2 tool calls deja ~5 checkpoints. Bonus: si el proceso muere a mitad
  del loop, se puede reanudar desde el último checkpoint.
- **Dónde**: en el backend del checkpointer, indexado por **`thread_id`** (el
  identificador de la conversación) + un id de checkpoint. `MemorySaver` = un
  diccionario en RAM (se pierde al reiniciar el kernel — perfecto para
  demos). En producción, mismo API con backend persistente (`SqliteSaver`,
  `PostgresSaver`): solo cambia el objeto que pasas a `compile()`.

### ¿CÓMO funciona el turno 2, mecánicamente?

```
invoke(mensaje_nuevo, thread_id="cliente-ana")
  1. el checkpointer CARGA el último checkpoint del thread "cliente-ana"
  2. add_messages (el reducer del estado) AGREGA el mensaje nuevo al final
  3. el node agente pasa TODA la lista al LLM → contexto completo
  4. cada node nuevo vuelve a guardar checkpoint
```

Es decir: la "memoria" no es magia ni un resumen — es **re-enviar el
historial completo** en cada turno. Consecuencia práctica: el contexto (y el
costo por token) **crece con la conversación**; en producción se recorta o
resume el historial pasado cierto tamaño.

- Cada `thread_id` es un universo aparte: la memoria de un cliente no se
  filtra a otro (lo probamos abajo).
- Esto es memoria de **corto plazo** (la conversación). La memoria de *largo
  plazo* (preferencias del usuario entre conversaciones) es otra pieza: un
  *store* consultable — buena extensión para explorar.
"""))

    cells.append(code(r"""
from langgraph.checkpoint.memory import MemorySaver

memoria = MemorySaver()
agente_con_memoria = construir_agente(llm, checkpointer=memoria)

hilo_ana = {"configurable": {"thread_id": "cliente-ana"}}

# Turno 1: preguntamos por un producto concreto.
r1 = agente_con_memoria.invoke(
    {"messages": [HumanMessage(content="¿Cuánto cuesta el teclado TM-1008?")]},
    config=hilo_ana)
print("— Turno 1:", r1["messages"][-1].content[:200], "\n")

# Turno 2: "¿y garantía?" — SIN repetir el SKU. El checkpoint lo recuerda.
r2 = agente_con_memoria.invoke(
    {"messages": [HumanMessage(content="¿Y cuántos meses de garantía tiene?")]},
    config=hilo_ana)
print("— Turno 2:", r2["messages"][-1].content[:300])
"""))

    cells.append(code(r"""
# La prueba de aislamiento: otro thread_id = otra conversación = amnesia.
hilo_luis = {"configurable": {"thread_id": "cliente-luis"}}
r3 = agente_con_memoria.invoke(
    {"messages": [HumanMessage(content="¿Y cuántos meses de garantía tiene?")]},
    config=hilo_luis)
print("— Mismo mensaje, thread nuevo:", r3["messages"][-1].content[:300])
"""))

    cells.append(code(r"""
# Autopsia de la memoria: ¿qué hay EXACTAMENTE guardado en el thread de Ana?
# get_state(config) devuelve el último checkpoint; .values es nuestro estado.
estado_ana = agente_con_memoria.get_state(hilo_ana)

print(f"Mensajes guardados en 'cliente-ana': "
      f"{len(estado_ana.values['messages'])}\n")
for i, m in enumerate(estado_ana.values["messages"], 1):
    tipo = type(m).__name__
    if isinstance(m, AIMessage) and m.tool_calls:
        detalle = "tool call → " + ", ".join(
            f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})"
            for tc in m.tool_calls)
    else:
        detalle = str(m.content).replace("\n", " ")[:90]
    print(f"  {i:>2}. {tipo:<14} {detalle}")

# Nota cómo los ToolMessage (las fichas, los pasajes del RAG) TAMBIÉN están
# en la memoria: por eso el turno 2 sabía el precio sin volver a buscar.

# Y el historial de checkpoints: una foto por node ejecutado en el thread.
n_checkpoints = sum(1 for _ in agente_con_memoria.get_state_history(hilo_ana))
print(f"\nCheckpoints acumulados en el thread: {n_checkpoints} "
      "(uno por node; sirven para reanudar o hacer time-travel)")

print("Mensajes en el hilo de Luis:",
      len(agente_con_memoria.get_state(hilo_luis).values["messages"]),
      "(su universo aparte)")
"""))

    cells.append(md(r"""
## 10. Todo junto: la app de Streamlit (con Docker)

Todo lo del notebook — agente + tools + reasoning + fuentes con foto +
memoria por thread — está empaquetado en una **app de Streamlit** en
`app/streamlit_app.py`, lista para correr en Docker:

```bash
# 1. Qdrant arriba y colección ingestada (notebook 01)
cd qdrant && docker compose up -d && cd ..

# 2. Construir y lanzar la app (lee las keys de tu .env)
cd app && docker compose up --build
```

→ abre <http://localhost:8501>

La app muestra en la barra lateral el **modelo** (elegible), el **toggle de
reasoning**, el **StateGraph** y el botón de *nueva conversación* (que
estrena `thread_id` — la memoria de la sección 9). Cada respuesta despliega
los **rounds** del loop ReAct (tool calls y resultados), el razonamiento del
modelo y las **fotos** de los productos consultados.

> El contenedor alcanza al Qdrant del host vía `host.docker.internal:6333`;
> las API keys entran por `env_file: ../.env`. Sin Docker también corre:
> `uv run streamlit run app/streamlit_app.py`.

## Resumen

- **LangGraph 101**: estado tipado con reducers, nodes = funciones que
  devuelven updates parciales, edges fijos y condicionales; `compile()`
  congela — extender = añadir al builder y recompilar.
- Un **agente** = LLM + tools + estado + loop ReAct: `agente ⇄ tools` con
  routing de `tools_condition`.
- El RAG del notebook 01 se volvió la tool `buscar_tecnomarket`, conviviendo
  con tools estructuradas (`detalle_producto`, `productos_por_presupuesto`),
  de "backend" (`estado_pedido`) y de cálculo (`calculadora`). La calidad de
  los **docstrings** es parte del prompt.
- `stream_mode="updates"` emite `{node: update}` por round — con eso armamos
  el playground que ilumina el StateGraph en vivo.
- **Reasoning** (`reasoning=True`) separa el borrador interno del modelo de
  su respuesta: auditable en `additional_kwargs["reasoning_content"]`.
- **Memoria** = checkpointer + `thread_id`: estado persistente por
  conversación sin cambiar el grafo.
- Todo junto corre como app en `app/` (Streamlit + Docker).

**Siguiente** → `03_mcp_rag_agentes.ipynb`: sacamos estas tools del notebook
y las servimos por **MCP**, el protocolo estándar para conectar tools a
*cualquier* host de LLM.
"""))

    return cells


# ===========================================================================
# Notebook 3 — MCP: conectar RAG + agentes vía Model Context Protocol
# ===========================================================================
def build_notebook_3() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# 03 — MCP: RAG + agentes conectados por el Model Context Protocol

Tercer notebook de la serie incremental — aquí **se junta todo**:

```
 notebook 01                    notebook 03                     notebook 02
┌────────────────────┐      ┌──────────────────────┐      ┌────────────────────┐
│ RAG multimodal     │      │  servidor MCP        │      │ agente LangGraph   │
│ Qdrant + Gemini    │ ◀────│  (FastMCP, stdio)    │◀─MCP─│ (cliente MCP)      │
│ + búsqueda híbrida │      │  herramientas:       │      │ LLM: Ollama cloud  │
└────────────────────┘      │  buscar / detalle    │      └────────────────────┘
                            └──────────────────────┘
```

## ¿Por qué un protocolo?

En el notebook 02, las herramientas viven **dentro** del proceso del agente:
funciones de Python enlazadas con `bind_tools`. Funciona, pero no escala
organizacionalmente: si mañana quieres esas mismas herramientas en Claude
Desktop, en Claude Code, en otro agente de otro equipo… reescribes la
integración cada vez ($M$ hosts × $N$ herramientas = $M{\times}N$ puentes).

El **Model Context Protocol (MCP)** estandariza ese enchufe: cada herramienta
se implementa **una vez** como *servidor MCP*, y cualquier *host* compatible
la consume ($M + N$ piezas). Conceptos:

- **Servidor MCP** — expone *tools* (funciones invocables), *resources*
  (datos leíbles por URI) y *prompts* (plantillas). El nuestro envolverá el
  RAG de TecnoMarket.
- **Host / cliente MCP** — la app donde vive el LLM (Claude Desktop, un agente
  LangGraph…); descubre las capacidades del servidor en tiempo de ejecución.
- **Transporte** — `stdio` (el host lanza el servidor como subproceso; ideal
  local) o HTTP (*streamable HTTP*; ideal remoto). Usaremos `stdio`.

> **Prerrequisitos:** notebook 01 ejecutado (colección en Qdrant), mismo
> `.env`. El servidor MCP reutiliza la API de Gemini para embeber consultas.
"""))

    cells.append(code(CONFIG_CELL))

    cells.append(md(r"""
## 1. El servidor MCP (FastMCP)

`FastMCP` (del SDK oficial `mcp`) convierte funciones de Python en un servidor
MCP con decoradores — la misma filosofía de `@tool` de LangChain, pero
sirviendo por protocolo en lugar de enlazar en proceso.

Escribimos el servidor a un archivo (`mcp/tecnomarket_server.py`) porque un
servidor stdio es un **proceso independiente**: el cliente lo lanzará con
`python tecnomarket_server.py`. Fíjate en que:

- expone **2 tools** (`buscar_catalogo`, `detalle_producto`) que replican las
  herramientas del notebook 02, ahora desacopladas del agente;
- expone **1 resource** (`politica://{nombre}`) — los documentos de políticas
  como datos direccionables por URI, sin pasar por búsqueda;
- carga el `.env` del módulo y habla con Qdrant + Gemini igual que el
  notebook 01 (búsqueda densa, para mantener el servidor corto).
"""))

    cells.append(code(r'''
%%writefile ../mcp/tecnomarket_server.py
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
'''))

    cells.append(md(r"""
## 2. Hablar con el servidor "a mano" (cliente MCP puro)

Antes de conectarlo a un agente, inspeccionamos el servidor con el cliente del
SDK, para ver el protocolo desnudo:

1. lanzar el servidor como subproceso (stdio),
2. `initialize` (negociación de capacidades),
3. `list_tools` — **descubrimiento**: el cliente no sabe de antemano qué hay,
4. `call_tool` y `read_resource`.

El SDK de MCP es **asíncrono**; en Jupyter podemos usar `await` directamente
en la celda (el notebook ya corre un event loop).
"""))

    cells.append(code(r"""
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVIDOR = StdioServerParameters(
    command=sys.executable,
    args=[str(ROOT / "mcp" / "tecnomarket_server.py")],
)

async with stdio_client(SERVIDOR) as (lectura, escritura):
    async with ClientSession(lectura, escritura) as sesion:
        await sesion.initialize()

        tools = await sesion.list_tools()
        print("Tools descubiertas:")
        for t in tools.tools:
            print(f"  - {t.name}: {t.description.splitlines()[0]}")

        r = await sesion.call_tool("buscar_catalogo",
                                   {"consulta": "¿cuánto cuesta el envío express?",
                                    "top_k": 2})
        print("\ncall_tool('buscar_catalogo') →\n", r.content[0].text[:400])

        recurso = await sesion.read_resource("politica://garantias")
        print("\nread_resource('politica://garantias') →",
              recurso.contents[0].text[:120], "…")
"""))

    cells.append(md(r"""
## 3. El agente del notebook 02, ahora con herramientas MCP

`langchain-mcp-adapters` traduce las tools MCP descubiertas a herramientas de
LangChain. El agente es el mismo de antes (LangGraph + LLM en Ollama cloud) —
pero ya **no importa las funciones**: las descubre por protocolo. Si mañana el
servidor agrega una herramienta, el agente la ve sin cambiar una línea.

Las herramientas MCP son asíncronas, así que usamos `ainvoke`/`astream`.
"""))

    cells.append(code(LLM_CELL))

    cells.append(code(r"""
from langchain_mcp_adapters.client import MultiServerMCPClient

cliente_mcp = MultiServerMCPClient({
    "tecnomarket": {
        "command": sys.executable,
        "args": [str(ROOT / "mcp" / "tecnomarket_server.py")],
        "transport": "stdio",
    },
})
tools_mcp = await cliente_mcp.get_tools()
print("Herramientas vía MCP:", [t.name for t in tools_mcp])
"""))

    cells.append(code(r"""
# El mismo grafo del notebook 02, ahora sobre herramientas descubiertas por MCP.
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class EstadoAgente(TypedDict):
    messages: Annotated[list, add_messages]

llm_con_tools = llm.bind_tools(tools_mcp)

async def nodo_agente(estado: EstadoAgente) -> dict:
    return {"messages": [await llm_con_tools.ainvoke(estado["messages"])]}

grafo = StateGraph(EstadoAgente)
grafo.add_node("agente", nodo_agente)
grafo.add_node("herramientas", ToolNode(tools_mcp))
grafo.add_edge(START, "agente")
grafo.add_conditional_edges("agente", tools_condition,
                            {"tools": "herramientas", END: END})
grafo.add_edge("herramientas", "agente")
agente = grafo.compile()
print("Agente con herramientas MCP compilado.")
"""))

    cells.append(code(r"""
pregunta = ("Quiero regalar el reloj Aviator: dame su SKU y precio, y dime "
            "qué cubre su garantía si se daña la correa.")

async for paso in agente.astream({"messages": [HumanMessage(content=pregunta)]},
                                 stream_mode="values"):
    paso["messages"][-1].pretty_print()
"""))

    cells.append(md(r"""
## 4. El mismo servidor, en otros hosts

Ese `tecnomarket_server.py` que acabamos de consumir desde LangGraph funciona
sin cambios en cualquier host MCP. Por ejemplo, en **Claude Desktop / Claude
Code** bastaría registrar en la configuración:

```json
{
  "mcpServers": {
    "tecnomarket": {
      "command": "uv",
      "args": ["run", "--project", "/ruta/a/module4-genai",
               "python", "/ruta/a/module4-genai/mcp/tecnomarket_server.py"]
    }
  }
}
```

…y Claude podría buscar en el catálogo de TecnoMarket en medio de cualquier
conversación. Una herramienta, N hosts: esa es la promesa del protocolo.

## Resumen de la serie

| Pieza | Notebook | Rol |
|---|---|---|
| Qdrant + `gemini-embedding-2` | 01 | memoria vectorial multimodal |
| Chunking + híbrida + métricas | 01 | calidad de la recuperación |
| LangChain + Ollama cloud | 01 | generación fundamentada |
| LangGraph | 02 | el loop ReAct; RAG como herramienta |
| MCP (FastMCP + adapters) | 03 | las herramientas como servicio estándar |

**Ideas para extender**: transporte HTTP para servir el RAG a hosts remotos;
más tools (inventario, órdenes); orquestar la re-ingesta del índice con
Airflow (introducido en el Módulo 1); evaluar el agente de punta a punta
(¿elige la herramienta correcta? ¿cuántas llamadas gasta?).
"""))

    return cells


def main() -> None:
    write_notebook(build_notebook_1(), NOTEBOOKS_DIR / "01_rag_multimodal.ipynb")
    write_notebook(build_notebook_2(), NOTEBOOKS_DIR / "02_agentes_langgraph.ipynb")
    write_notebook(build_notebook_3(), NOTEBOOKS_DIR / "03_mcp_rag_agentes.ipynb")


if __name__ == "__main__":
    main()
