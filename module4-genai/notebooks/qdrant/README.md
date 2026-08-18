# Qdrant — local vector database

[Qdrant](https://qdrant.tech/) is the vector database we use throughout this
module to store embeddings and run similarity search (dense + sparse / hybrid).

## Start

```bash
cd qdrant
docker compose up -d
```

This launches a single Qdrant node with a persistent named volume
(`module4_qdrant_storage`), so your collections survive container restarts.

Ports exposed:

| Port | Protocol | Use |
|------|----------|-----|
| 6333 | HTTP / REST | Client API + web dashboard |
| 6334 | gRPC | Faster client API (optional) |

## Dashboard

Once it's healthy, open the built-in web UI:

<http://localhost:6333/dashboard>

You can browse collections, inspect points/payloads, and run queries from the
console there.

## Health check

```bash
docker compose ps          # STATUS should show "healthy"
curl http://localhost:6333/readyz
```

## Teardown

```bash
docker compose down              # stop + remove the container (keeps data)
docker compose down -v           # also delete the named volume (wipes data)
```

## Next steps

With Qdrant running, ingest the sample docs and try the RAG pipeline:

```bash
cd ..
python rag/ingest.py
streamlit run rag/streamlit_app.py
```
