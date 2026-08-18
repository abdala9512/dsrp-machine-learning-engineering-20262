"""TecnoMarket — agente LangGraph en Streamlit (Módulo 4, notebook 02).

Todo el notebook 02 en una app: agente ReAct con tools (RAG híbrido sobre
Qdrant, ficha por SKU, filtro por presupuesto, tracking de pedidos,
calculadora), reasoning opcional, fuentes con foto y memoria por thread.

Local:   uv run streamlit run app/streamlit_app.py
Docker:  cd app && docker compose up --build     → http://localhost:8501

─────────────────────────────────────────────────────────────────────────────
CÓMO FUNCIONA STREAMLIT (el modelo mental para leer este archivo)
─────────────────────────────────────────────────────────────────────────────
Streamlit NO es un framework de callbacks: es un *script* que se
**re-ejecuta completo, de arriba a abajo, en cada interacción** (cada clic,
cada mensaje del chat). De ese modelo salen las tres piezas que verás
repetidas aquí:

1. `@st.cache_resource` — un **decorador**: una función que envuelve a otra
   y le agrega comportamiento. Aquí el comportamiento es *memoización*: la
   primera llamada ejecuta la función y guarda el resultado; las siguientes
   (mismos argumentos) devuelven lo guardado SIN ejecutar. Sin esto,
   reconectaríamos con Qdrant y recompilaríamos el grafo en cada tecla.

2. `st.session_state` — un diccionario que **sobrevive entre re-ejecuciones**
   (por pestaña del navegador). Todo lo que queramos "recordar" entre
   interacciones (el historial del chat, el thread_id) vive ahí; las
   variables normales de Python mueren al final de cada corrida del script.

3. El **orden del archivo es el orden de la página**: primero se pinta el
   sidebar, luego el historial, y al final se procesa el mensaje nuevo.

Estructura del archivo:

    §1  Configuración (rutas, .env, constantes)
    §2  Estado del agente (TypedDict + reducer, a nivel de módulo)
    §3  cargar_rag()        → conexiones a Qdrant/Gemini + búsqueda híbrida
    §4  construir_agente()  → LLM + tools + StateGraph compilado con memoria
    §5  Sidebar (controles)
    §6  Helpers de UI (galería de fotos)
    §7  Render del historial
    §8  El turno nuevo (streaming del loop ReAct)
"""

from __future__ import annotations

# ============================================================================
# §1. CONFIGURACIÓN
# ============================================================================
import hashlib
import json
import time
import uuid
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph.message import add_messages

# ROOT = la raíz del módulo (module4-genai/), tanto si corres local como en
# Docker (donde este archivo vive en /srv/app/ y el catálogo en /srv/rag/).
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # en Docker no existe: las keys entran por compose

import os  # noqa: E402  (después de load_dotenv para leer el .env ya cargado)

GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "tecnomarket")
CATALOG_DIR = ROOT / "rag" / "catalog"

# El modelo se elige en la UI (sidebar), igual que en los notebooks se elige
# en una celda: es un knob del experimento, no un secreto del .env.
MODELOS = ["minimax-m3:cloud", "gpt-oss:120b-cloud", "kimi-k3:cloud"]

# Primera llamada a Streamlit del script: define título/ícono/layout de la
# página. Debe ir antes que cualquier otro st.* que pinte algo.
st.set_page_config(page_title="TecnoMarket · Agente", page_icon="🛒", layout="wide")


# ============================================================================
# §2. EL ESTADO DEL AGENTE (mismo patrón del notebook 02, sección 5)
# ============================================================================
class EstadoAgente(TypedDict):
    """El estado compartido del StateGraph: una sola clave, `messages`.

    `Annotated[list, add_messages]` adjunta un **reducer** al campo: cuando
    un node devuelve mensajes nuevos, add_messages los AGREGA a la lista en
    lugar de sobrescribirla — así la conversación crece node a node.

    Nota de implementación: la clase vive a nivel de MÓDULO (no dentro de
    construir_agente) porque, con `from __future__ import annotations`, las
    anotaciones son strings que LangGraph evalúa en los globals del módulo —
    `add_messages` tiene que ser visible aquí.
    """

    messages: Annotated[list, add_messages]


# ============================================================================
# §3. NÚCLEO RAG — conexiones y búsqueda híbrida (notebook 01, compactado)
# ============================================================================
# El decorador @st.cache_resource memoiza la función: se ejecuta UNA vez y el
# resultado (conexiones, índice BM25, catálogo) se reutiliza en cada
# re-ejecución del script y entre pestañas. "resource" (vs @st.cache_data)
# indica que el valor es un objeto vivo no serializable (clientes de red).
@st.cache_resource(show_spinner="Conectando con Qdrant y Gemini…")
def cargar_rag():
    """Prepara el motor de búsqueda y devuelve tres piezas:

    - ``buscar_hibrido(consulta, top_k)`` — función de búsqueda híbrida
      (densa por embeddings + léxica BM25, fusionadas con RRF).
    - ``products`` — la lista completa del catálogo (products.json).
    - ``prod_por_sku`` — índice {sku: producto} para lookups exactos.

    Los imports pesados van DENTRO de la función: solo se pagan en la
    primera ejecución (después el decorador devuelve el resultado cacheado
    sin entrar aquí).
    """
    from google import genai
    from google.genai import types
    from qdrant_client import QdrantClient
    from rank_bm25 import BM25Okapi

    # --- Conexiones ---
    gclient = genai.Client()  # lee GEMINI_API_KEY del entorno
    qdrant = QdrantClient(url=QDRANT_URL)
    if not qdrant.collection_exists(COLLECTION):
        # st.stop() corta la ejecución del script aquí: la página muestra
        # solo el error, sin romper con un traceback.
        st.error(
            f"La colección '{COLLECTION}' no existe en {QDRANT_URL}. "
            "Ejecuta el notebook 01 (ingesta) con Qdrant arriba."
        )
        st.stop()

    # --- Corpus léxico: reconstruimos BM25 leyendo los payloads de Qdrant ---
    # (la colección es la fuente de la verdad; no dependemos del notebook 01
    # en memoria). scroll() pagina todos los puntos con sus payloads.
    puntos, _ = qdrant.scroll(collection_name=COLLECTION, limit=1000, with_payload=True)
    payload_por_ref = {p.payload["ref"]: p.payload for p in puntos}
    corpus = [(ref, pl["texto"]) for ref, pl in payload_por_ref.items() if pl["texto"]]
    refs_bm25 = [ref for ref, _ in corpus]
    bm25 = BM25Okapi([t.lower().split() for _, t in corpus])

    # lru_cache = memoización en memoria por argumento: la misma consulta no
    # paga la API de embeddings dos veces (equivale al cache del notebook).
    @lru_cache(maxsize=512)
    def embed_consulta(texto: str):
        """Embebe la consulta con Gemini (formato asimétrico de retrieval),
        con reintentos y backoff exponencial para los límites de tasa (429)."""
        for intento in range(5):
            try:
                r = gclient.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=f"task: search result | query: {texto}",
                    config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
                )
                return list(r.embeddings[0].values)
            except Exception:
                time.sleep(2**intento)
        raise RuntimeError("API de embeddings sin respuesta.")

    def buscar_hibrido(consulta: str, top_k: int = 4, pool: int = 15, rrf_k: int = 60):
        """Búsqueda híbrida (notebook 01, sección 7).

        1. Rama DENSA: embedding de la consulta → top-`pool` vecinos en Qdrant.
        2. Rama LÉXICA: puntuación BM25 de la consulta contra el corpus.
        3. FUSIÓN RRF: cada resultado suma 1/(rrf_k + posición) por cada
           ranking donde aparece; se devuelven los `top_k` mejores payloads.
        """
        densos = qdrant.query_points(
            collection_name=COLLECTION, query=embed_consulta(consulta),
            limit=pool, with_payload=True,
        ).points
        scores = bm25.get_scores(consulta.lower().split())
        orden_lex = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        fusion: dict[str, float] = {}
        for rank, h in enumerate(densos):
            fusion[h.payload["ref"]] = fusion.get(h.payload["ref"], 0) + 1 / (rrf_k + rank)
        for rank, i in enumerate(orden_lex[:pool]):
            if scores[i] > 0:
                ref = refs_bm25[i]
                fusion[ref] = fusion.get(ref, 0) + 1 / (rrf_k + rank)
        mejores = sorted(fusion, key=fusion.get, reverse=True)[:top_k]
        return [payload_por_ref[ref] for ref in mejores]

    products = json.loads((CATALOG_DIR / "products.json").read_text(encoding="utf-8"))
    return buscar_hibrido, products, {p["sku"]: p for p in products}


# Desempacamos las tres piezas a nivel de módulo (MAYÚSCULAS = "constantes"
# del script; en realidad se re-asignan en cada corrida, pero desde el cache).
BUSCAR_HIBRIDO, PRODUCTS, PROD_POR_SKU = cargar_rag()


# ============================================================================
# §4. EL AGENTE — LLM + tools + StateGraph (notebook 02, secciones 3-5)
# ============================================================================
# También cacheado, pero OJO: la clave del cache son los ARGUMENTOS
# (modelo, reasoning). Cambiar el modelo en el sidebar = otra clave = se
# construye otro agente (con su propia memoria MemorySaver).
@st.cache_resource(show_spinner="Conectando el LLM y compilando el grafo…")
def construir_agente(modelo: str, reasoning: bool):
    """Construye el agente completo y devuelve una tupla de tres:

    - ``agente`` — el StateGraph compilado con checkpointer (memoria).
    - ``modelo_activo`` — el modelo que realmente respondió (puede ser el
      fallback si el elegido no está disponible en el plan).
    - ``fuentes_recientes`` — lista mutable que las tools van llenando con
      los SKUs consultados; la UI la lee para pintar las fotos (§8).

    Parámetros
    ----------
    modelo : str
        Modelo de Ollama cloud elegido en el sidebar.
    reasoning : bool
        Si True, pide al modelo sus tokens de razonamiento (thinking);
        llegan en ``additional_kwargs["reasoning_content"]`` de cada
        AIMessage, separados de la respuesta.
    """
    from langchain_core.tools import tool
    from langchain_ollama import ChatOllama
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
    from ollama import Client as OllamaClient

    headers = {"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]}

    # --- Probe con fallback (mismo patrón de los notebooks) ---------------
    # Antes de comprometernos con un modelo, le mandamos un ping de 1 token.
    # Si falla (p. ej. HTTP 402: modelo de pago sin saldo), probamos el
    # fallback del plan gratuito.
    probe = OllamaClient(host=OLLAMA_HOST, headers=headers)
    llm = None
    for m in [modelo, "gpt-oss:120b-cloud"]:
        try:
            probe.chat(model=m, messages=[{"role": "user", "content": "ok"}],
                       options={"num_predict": 1})
        except Exception:
            continue  # este modelo no responde → probar el siguiente
        llm = ChatOllama(model=m, base_url=OLLAMA_HOST,
                         client_kwargs={"headers": headers},
                         temperature=0.1,
                         reasoning=reasoning or None)  # None = default (off)
        break
    if llm is None:
        st.error("Ningún modelo de Ollama cloud respondió; revisa OLLAMA_API_KEY.")
        st.stop()

    # Las tools de más abajo son "closures": funciones definidas DENTRO de
    # esta función, que capturan variables locales como esta lista. Las
    # tools la llenan; la UI la vacía y la lee en cada turno (§8).
    fuentes_recientes: list[dict] = []

    # --- Las 5 tools (notebook 02, sección 3) -----------------------------
    # @tool es un decorador de LangChain: convierte una función de Python en
    # una herramienta invocable por el LLM. El DOCSTRING y los TYPE HINTS no
    # son adorno: se convierten en el JSON Schema que el modelo lee para
    # decidir cuándo y cómo llamar cada tool. Escríbelos para el modelo.

    @tool
    def buscar_tecnomarket(consulta: str) -> str:
        'Busca en la base de conocimiento de TecnoMarket (políticas de envíos, devoluciones y garantías, y catálogo de productos). Recibe una consulta en lenguaje natural y devuelve los pasajes más relevantes.'
        resultados = BUSCAR_HIBRIDO(consulta, top_k=4)
        fuentes_recientes.extend(r for r in resultados if r.get("sku"))
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
        # Lookup exacto en el diccionario del catálogo — sin embeddings:
        # no todo es búsqueda semántica.
        p = PROD_POR_SKU.get(sku.strip().upper())
        if not p:
            return f"No existe el SKU {sku}."
        fuentes_recientes.append({"sku": p["sku"]})
        return json.dumps({k: p[k] for k in ("sku", "nombre", "categoria",
                                             "precio", "descripcion")},
                          ensure_ascii=False)

    @tool
    def productos_por_presupuesto(categoria: str, presupuesto: float) -> str:
        'Lista los productos de TecnoMarket de una categoría (p. ej. "audifonos", "zapatillas", "relojes", "cafeteras") con precio menor o igual al presupuesto dado, ordenados del más barato al más caro.'
        # Filtro ESTRUCTURADO (categoría + precio): el modelo extrae los dos
        # argumentos de la frase del usuario, el código hace el resto.
        cat = categoria.strip().lower()
        encontrados = sorted(
            (p for p in PRODUCTS if p["categoria"] == cat and p["precio"] <= presupuesto),
            key=lambda p: p["precio"])[:5]
        if not encontrados:
            # Devolver las categorías válidas AYUDA al modelo a auto-corregir
            # el argumento en el siguiente round (p. ej. "audífonos" → "audifonos").
            categorias = sorted({p["categoria"] for p in PRODUCTS})
            return (f"Nada en '{categoria}' por ≤ {presupuesto}. "
                    f"Categorías disponibles: {', '.join(categorias)}.")
        fuentes_recientes.extend({"sku": p["sku"]} for p in encontrados)
        return json.dumps([{k: p[k] for k in ("sku", "nombre", "precio")}
                           for p in encontrados], ensure_ascii=False)

    @tool
    def estado_pedido(numero_pedido: str) -> str:
        'Consulta el estado de un pedido de TecnoMarket dado su número (formato PED-XXXX): estado actual, transportadora y fecha estimada de entrega.'
        # Simulación determinista: el hash del número decide el estado (el
        # mismo pedido siempre responde lo mismo). En producción, esta tool
        # llamaría al API interno de órdenes — al agente le da igual: solo
        # ve el docstring y el resultado.
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
        # Whitelist de caracteres antes del eval: solo aritmética, nada de
        # nombres de Python → un eval seguro para este caso acotado.
        if not set(expresion) <= set("0123456789+-*/(). "):
            return "Error: solo se admite aritmética básica."
        try:
            return str(eval(expresion, {"__builtins__": {}}, {}))
        except Exception as exc:
            return f"Error: {exc}"

    tools = [buscar_tecnomarket, detalle_producto, productos_por_presupuesto,
             estado_pedido, calculadora]

    # --- El StateGraph (notebook 02, sección 5) ---------------------------
    # bind_tools adjunta los JSON Schemas de las tools a cada llamada al
    # modelo: eso habilita el tool calling nativo.
    llm_con_tools = llm.bind_tools(tools)

    # El system prompt NO aparece solo en un agente: el modelo ve exactamente
    # los mensajes que le pasamos. Lo anteponemos en cada llamada, sin
    # guardarlo en el estado (no se duplica en la memoria y se puede cambiar
    # sin invalidar los checkpoints).
    from langchain_core.messages import SystemMessage

    system_prompt = SystemMessage(content=(
        "Eres el asistente de la tienda TecnoMarket. Responde en español, "
        "breve y amable. Usa las tools para consultar catálogo, políticas y "
        "pedidos: no inventes SKUs, precios ni plazos. Si no encuentras la "
        "información, dilo claramente."))

    def nodo_agente(estado: EstadoAgente) -> dict:
        """Node 'agente': system prompt + TODA la conversación al LLM;
        devuelve su respuesta como update parcial (el reducer add_messages
        la agrega al estado)."""
        return {"messages": [llm_con_tools.invoke([system_prompt] + estado["messages"])]}

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("tools", ToolNode(tools))       # ejecuta los tool calls
    grafo.add_edge(START, "agente")                # edge fijo: entrada
    grafo.add_conditional_edges(                   # edge condicional:
        "agente", tools_condition,                 # ¿el modelo pidió tools?
        {"tools": "tools", END: END})              # sí → tools ; no → fin
    grafo.add_edge("tools", "agente")              # cierra el loop ReAct

    # compile(checkpointer=...) activa la MEMORIA (notebook 02, sección 9):
    # MemorySaver guarda un checkpoint del estado tras cada node, indexado
    # por thread_id. Mismo thread_id → la conversación continúa.
    agente = grafo.compile(checkpointer=MemorySaver())
    return agente, llm.model, fuentes_recientes


# ============================================================================
# §5. SIDEBAR — los controles de la app
# ============================================================================
# `with st.sidebar:` = "todo lo que pinte dentro de este bloque va a la
# barra lateral" (un context manager de layout).
with st.sidebar:
    st.title("🛒 TecnoMarket")
    st.caption("Agente LangGraph · Módulo 4, notebook 02")

    # Los widgets devuelven su valor actual en cada re-ejecución del script.
    modelo_elegido = st.selectbox("Modelo (Ollama cloud)", MODELOS)
    con_reasoning = st.toggle(
        "🧠 Reasoning", value=False,
        help="Pide los tokens de razonamiento del modelo (thinking). "
             "Más lento; visible por round.")

    # Nueva conversación = borrar thread_id + historial del session_state y
    # re-ejecutar el script (st.rerun). Abajo (§7) se regeneran vacíos.
    if st.button("🔄 Nueva conversación", use_container_width=True):
        st.session_state.pop("thread_id", None)
        st.session_state.pop("historial", None)
        st.rerun()

    st.divider()
    st.subheader("StateGraph")
    st.code("START → agente ⇄ tools\n         agente → END", language=None)
    st.caption("Loop ReAct: el edge condicional pregunta "
               "«¿el modelo pidió tools?».")
    st.divider()
    st.caption(f"Qdrant: `{QDRANT_URL}`\nColección: `{COLLECTION}`")

# Con los valores de los widgets, pedimos el agente (del cache si ya existe
# para esta combinación modelo+reasoning; construido si es la primera vez).
agente, modelo_activo, FUENTES = construir_agente(modelo_elegido, con_reasoning)

# session_state: memoria de la PESTAÑA (ver cabecera). Inicializamos las dos
# claves solo si no existen todavía (primer render o tras "Nueva conversación").
if "thread_id" not in st.session_state:
    # uuid4 = identificador aleatorio único → cada conversación es un thread
    # distinto para el checkpointer del agente.
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
if "historial" not in st.session_state:
    # Lo que la UI re-pinta en cada corrida: lista de turnos
    # (rol, contenido_markdown, extras) — extras guarda rounds y SKUs.
    st.session_state.historial = []

st.markdown(f"##### Conversación `{st.session_state.thread_id}` · modelo `{modelo_activo}`")


# ============================================================================
# §6. HELPERS DE UI
# ============================================================================
def galeria_fuentes(skus: list[str]):
    """Pinta las fotos de los productos consultados por el agente.

    Recibe la lista de SKUs que las tools registraron durante el turno,
    quita duplicados preservando el orden, y muestra hasta 5 tarjetas
    (foto + sku + nombre + precio) en columnas.
    """
    productos, vistos = [], set()
    for sku in skus:
        p = PROD_POR_SKU.get(sku)
        if p and sku not in vistos:
            vistos.add(sku)
            productos.append(p)
    if not productos:
        return
    st.caption("🖼 Productos consultados por el agente:")
    cols = st.columns(min(len(productos), 5))
    for col, p in zip(cols, productos):
        with col:
            ruta = CATALOG_DIR / "images" / p["imagen"]
            if ruta.exists():
                st.image(str(ruta), use_container_width=True)
            st.caption(f"**{p['sku']}** · {p['nombre']} · ${p['precio']:,.0f}")


# ============================================================================
# §7. RENDER DEL HISTORIAL
# ============================================================================
# Recuerda: el script COMPLETO corre en cada interacción. El chat "persiste"
# porque re-pintamos aquí todos los turnos guardados en session_state.
for rol, contenido, extras in st.session_state.historial:
    with st.chat_message(rol):                     # burbuja de chat (user/assistant)
        st.markdown(contenido)
        if extras.get("rounds"):                   # traza ReAct plegada
            with st.expander(f"⚙️ {len(extras['rounds'])} rounds del loop ReAct"):
                for linea in extras["rounds"]:
                    st.markdown(linea)
        if extras.get("skus"):                     # fotos de las fuentes
            galeria_fuentes(extras["skus"])


# ============================================================================
# §8. EL TURNO NUEVO — streaming del loop ReAct
# ============================================================================
# st.chat_input devuelve None en la mayoría de corridas; devuelve el texto
# solo en la corrida disparada por un mensaje nuevo. El walrus (:=) asigna
# y evalúa en la misma línea: el bloque solo corre cuando HAY pregunta.
if pregunta := st.chat_input("Pregunta por productos, envíos, garantías, tu pedido…"):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    # 1. Registrar y pintar el turno del usuario.
    st.session_state.historial.append(("user", pregunta, {}))
    with st.chat_message("user"):
        st.markdown(pregunta)

    # 2. Preparar la corrida: vaciar el registro de fuentes y armar el
    #    config con el thread_id (así el checkpointer carga/continúa ESTA
    #    conversación — la memoria de la sección 9 del notebook).
    FUENTES.clear()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    rounds: list[str] = []   # traza para re-pintar el turno en el futuro (§7)
    respuesta = ""

    # 3. Correr el agente EN STREAMING y narrar cada round.
    with st.chat_message("assistant"):
        # st.status = panel plegable con estado (spinner → check al final).
        with st.status("El agente está trabajando…", expanded=True) as status:
            n = 0
            # stream_mode="updates" emite {nombre_del_node: update} tras
            # CADA node — la misma señal que usa el playground del notebook
            # para iluminar el StateGraph.
            for paso in agente.stream(
                    {"messages": [HumanMessage(content=pregunta)]},
                    config=config, stream_mode="updates"):
                nodo, update = next(iter(paso.items()))
                n += 1
                st.write(f"**Round {n} · node `{nodo}`**")
                for msg in update.get("messages", []):
                    if isinstance(msg, AIMessage):
                        # a) razonamiento (si el toggle 🧠 está activo)
                        r = msg.additional_kwargs.get("reasoning_content")
                        if r:
                            st.markdown(f"🧠 *{r[:600]}…*" if len(r) > 600 else f"🧠 *{r}*")
                            rounds.append(f"**round {n}** 🧠 *{r[:300]}…*")
                        # b) tool calls que el modelo decidió hacer
                        for tc in msg.tool_calls:
                            linea = (f"🛠 `{tc['name']}"
                                     f"({json.dumps(tc['args'], ensure_ascii=False)})`")
                            st.markdown(linea)
                            rounds.append(f"**round {n}** {linea}")
                        # c) texto final (cuando ya no pide tools)
                        if msg.content:
                            respuesta = msg.content
                    elif isinstance(msg, ToolMessage):
                        # resultado que la tool devolvió al modelo
                        corto = str(msg.content)[:250]
                        st.markdown(f"📦 `{msg.name}` → {corto}…")
                        rounds.append(f"**round {n}** 📦 `{msg.name}` → {corto}…")
            status.update(label=f"Listo en {n} rounds", state="complete",
                          expanded=False)

        # 4. La respuesta final + las fotos de las fuentes del turno.
        st.markdown(respuesta or "*(sin respuesta)*")
        skus = [f["sku"] for f in FUENTES if f.get("sku")]
        galeria_fuentes(skus)

    # 5. Guardar el turno completo en session_state para que §7 pueda
    #    re-pintarlo en las próximas corridas del script.
    st.session_state.historial.append(
        ("assistant", respuesta, {"rounds": rounds, "skus": skus}))
