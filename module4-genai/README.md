# Módulo 4 — RAG multimodal, Agentes y MCP

Este módulo construye, en **tres notebooks incrementales**, los bloques de las
aplicaciones modernas de IA Generativa:

| Notebook | Qué construye |
|---|---|
| `01_rag_multimodal.ipynb` | Un pipeline de **RAG multimodal** completo y evaluado: Qdrant + `gemini-embedding-2` + LLM en Ollama cloud |
| `02_agentes_langgraph.ipynb` | Un **agente LangGraph** que usa ese RAG como *tool* (RAG agéntico): LangGraph 101 (nodes/edges), 5 tools, playground interactivo del loop ReAct, reasoning, fuentes con foto y memoria por `thread_id` |
| `03_mcp_rag_agentes.ipynb` | Los mismos componentes conectados vía **MCP** (Model Context Protocol) |

## El stack

- **Embeddings**: `gemini-embedding-2` (API de Gemini) — **multimodal nativo**:
  texto e imágenes se proyectan al *mismo* espacio vectorial (768 dims vía MRL),
  así que una consulta escrita recupera directamente fotos de producto.
- **Base vectorial**: **Qdrant** en Docker (`qdrant/docker-compose.yml`),
  colección con payloads y filtrado por modalidad.
- **LLM**: servido por **Ollama cloud** vía `langchain-ollama`. El modelo se
  elige **en una celda del notebook** (variable `OLLAMA_MODEL`; por defecto
  `minimax-m3:cloud`, con `kimi-k3:cloud` como opción multimodal de pago). Si
  el elegido no está disponible en tu plan, los notebooks degradan
  automáticamente a `gpt-oss:120b-cloud` (plan gratuito).
- **Orquestación del pipeline**: LangChain (splitters, prompts, LCEL) y
  LangGraph (agentes); `langchain-mcp-adapters` + FastMCP para el notebook 03.

## El caso de uso

**TecnoMarket**, una tienda ficticia (`rag/catalog/`): **~100 productos** en
~26 categorías, cada uno con fotografía (CC, de Wikimedia Commons) +
descripción con SKU, y 3 documentos de políticas (envíos, devoluciones,
garantías). El corpus mezcla texto largo (chunking), texto corto con términos
exactos (los SKU hacen brillar a BM25 en la búsqueda híbrida) e imágenes
(recuperación cross-modal), con suficientes distractores para que las métricas
de recuperación signifiquen algo.

El notebook 01 cubre además: estrategia de **chunking**
(`RecursiveCharacterTextSplitter`), **búsqueda híbrida** (densa + BM25
fusionadas con RRF), **evaluación de la recuperación** con *hit rate* y *MRR*
(explicadas paso a paso y medidas sobre un evalset anotado) y **widgets**
(`ipywidgets`) para validar resultados: búsqueda por texto, **búsqueda por
imagen** (similitud visual, incluso subiendo tu propia foto) y un panel de
métricas en vivo. Los embeddings de la ingesta se **cachean en disco**
(`rag/catalog/embeddings_cache.npz`, incluido en el repo) para no gastar cuota
de API en cada re-ejecución.

## Contenido

```
module4-genai/
├── notebooks/
│   ├── 01_rag_multimodal.ipynb      RAG multimodal de punta a punta (prioridad)
│   ├── 02_agentes_langgraph.ipynb   agente con el RAG como herramienta
│   └── 03_mcp_rag_agentes.ipynb     RAG + agente + MCP conectados
├── rag/catalog/                     dataset TecnoMarket
│   ├── products.json                ~100 productos (sku, descripción, imagen…)
│   ├── docs/                        políticas: envíos, devoluciones, garantías
│   ├── images/                      fotos de producto (+ ATTRIBUTIONS.md)
│   └── embeddings_cache.npz         cache de embeddings (evita re-pagar la API)
├── mcp/tecnomarket_server.py        servidor MCP (el notebook 03 lo regenera)
├── app/                             app Streamlit del agente (notebook 02)
│   ├── streamlit_app.py             chat + rounds ReAct + reasoning + fotos + memoria
│   ├── Dockerfile                   imagen de la app
│   └── docker-compose.yml           cd app && docker compose up --build
├── qdrant/docker-compose.yml        despliegue de Qdrant
└── _build_notebooks.py              genera los notebooks (no editar los .ipynb a mano)
```

## Cómo ejecutar

1. **Dependencias y credenciales**

   ```bash
   cd module4-genai
   uv sync
   cp .env.example .env    # edita: GEMINI_API_KEY y OLLAMA_API_KEY
   ```

   - `GEMINI_API_KEY`: <https://aistudio.google.com/apikey> (embeddings).
   - `OLLAMA_API_KEY`: <https://ollama.com/settings/keys> (LLM).
   - El modelo del LLM se elige en la celda `OLLAMA_MODEL` de cada notebook
     (no en el `.env`). `kimi-k3:cloud` requiere saldo de *extra usage*; sin
     él, los notebooks caen a `gpt-oss:120b-cloud` automáticamente.

2. **Desplegar Qdrant**

   ```bash
   cd qdrant && docker compose up -d && cd ..
   ```
   Dashboard: <http://localhost:6333/dashboard>

3. **Recorrer los notebooks en orden**

   ```bash
   uv run jupyter lab notebooks/
   ```

   El 01 crea e ingesta la colección (los 02 y 03 la reutilizan — ejecútalo
   primero). La ingesta usa el cache de embeddings incluido en el repo: solo
   llama a la API para contenido nuevo o modificado.

4. **(Opcional) La app del agente en Docker**

   Con Qdrant arriba y la colección ingestada:

   ```bash
   cd app && docker compose up --build
   ```

   → <http://localhost:8501>: chat con el agente del notebook 02 (rounds del
   loop ReAct, reasoning opcional, fotos de los productos consultados y
   memoria por conversación). Sin Docker:
   `uv run streamlit run app/streamlit_app.py`.

## Variables de entorno (`.env`)

| Variable | Por defecto | Propósito |
|----------|-------------|-----------|
| `GEMINI_API_KEY` | *(vacío)* | Embeddings multimodales (obligatoria). |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Modelo de embeddings. |
| `EMBEDDING_DIM` | `768` | Dimensión de salida (MRL: 768/1536/3072). |
| `OLLAMA_API_KEY` | *(vacío)* | LLM en Ollama cloud (obligatoria). |
| `OLLAMA_HOST` | `https://ollama.com` | Endpoint de Ollama cloud. |
| `QDRANT_URL` | `http://localhost:6333` | Endpoint de Qdrant. |
| `QDRANT_COLLECTION` | `tecnomarket` | Nombre de la colección. |

> **Nota sobre notebooks:** los `.ipynb` se generan con
> `uv run python _build_notebooks.py`. Si quieres cambiar el contenido, edita
> `_build_notebooks.py` y regenera.
