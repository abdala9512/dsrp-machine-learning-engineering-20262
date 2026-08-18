# App Streamlit del agente TecnoMarket (Módulo 4, notebook 02).
# Construir desde la RAÍZ del módulo (docker-compose ya lo hace):
#   docker build -f app/Dockerfile -t tecnomarket-agente .
FROM python:3.12-slim

WORKDIR /srv

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Catálogo (fotos + products.json) y la app. El .env NO se copia a la
# imagen: las keys entran en runtime (env_file del compose).
COPY rag/catalog ./rag/catalog
COPY app/streamlit_app.py ./app/streamlit_app.py

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
