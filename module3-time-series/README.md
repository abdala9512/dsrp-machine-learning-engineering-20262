# Módulo 3 — Series de Tiempo (+ MLflow)

Curso de ML Avanzado. Este módulo cubre el conjunto de herramientas completo de
series de tiempo: entender *de qué está hecha una serie*, el pronóstico
estadístico clásico (suavizamiento exponencial, ARIMA), el pronóstico con
machine learning moderno, los **ensambles de pronósticos** y el descubrimiento
no supervisado de patrones — todo sobre **un mismo dataset real** y con
**tracking de experimentos en MLflow**, igual que en el Módulo 2.

## El dataset del módulo

Todos los notebooks usan
[Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
(UCI ML Repository #235): mediciones eléctricas de una vivienda en Sceaux
(Francia) cada **minuto**, de diciembre de 2006 a noviembre de 2010
(~2M filas, ~1.25% de valores faltantes). Trabajamos con la potencia activa
global (kW) agregada a resolución **diaria** (pronóstico) y **horaria**
(perfiles de carga).

La primera ejecución descarga el zip (~20 MB) y deja cachés CSV en `data/`;
las siguientes leen el caché. Sin internet, los notebooks caen a un respaldo
sintético con estacionalidad semanal + anual para poder seguir la clase.

## Objetivos

Al terminar este módulo deberías ser capaz de:

- Descomponer una serie en tendencia / estacionalidad / residual (clásica y
  **STL**) y diagnosticar la **estacionariedad** (móviles, ACF/PACF, ADF).
- Establecer **baselines** (naive, naive estacional) y evaluar cualquier
  pronóstico con **MSE, RMSE, MAE, MAPE y sMAPE**.
- Ajustar y validar **suavizamiento exponencial** (SES, Holt, Holt-Winters) y
  **ARIMA / SARIMA** (diagnóstico de residuos, intervalos de confianza).
- Construir **variables para series de tiempo** (rezagos, ventanas móviles,
  calendario, Fourier) sin fuga temporal, y validar con `TimeSeriesSplit`.
- Pronosticar con **XGBoost** (multi-paso recursivo vs directo).
- Combinar modelos en **ensambles de pronósticos** (media, mediana, pesos por
  error inverso, stacking).
- Agrupar series por **forma** con **DTW** e interpretar los clusters.
- Comprimir perfiles de carga con **NMF** (factorización no negativa) en
  patrones horarios interpretables y agrupar en el espacio reducido.
- Registrar cada experimento en **MLflow** y promover los mejores modelos al
  **Model Registry**.

## Los seis notebooks

| # | Notebook | Qué enseña |
|---|----------|------------|
| 1 | `notebooks/01_decomposition.ipynb` | El dataset; componentes de una serie; aditivo vs multiplicativo; `seasonal_decompose` y **STL**; estacionariedad (móviles, ACF/PACF, ADF); las **5 métricas** del módulo y los **baselines** (naive, naive estacional, media móvil) registrados en MLflow. |
| 2 | `notebooks/02_arima.ipynb` | **Suavizamiento exponencial**: SES → Holt (amortiguado) → **Holt-Winters** con su matemática; **ARIMA/SARIMA** (operador de rezago, elección de órdenes vía ACF/PACF, diagnósticos Ljung-Box + Q-Q, pronóstico con IC); `auto_arima` opcional; comparación de todos + **Model Registry**. |
| 3 | `notebooks/03_feature_engineering.ipynb` | El pronóstico como regresión supervisada; **rezagos**, **ventanas móviles** (y la trampa de fuga del `shift(1)`), **calendario** y **Fourier**; `TimeSeriesSplit`; sanity-check con Ridge; exporta la matriz de variables. |
| 4 | `notebooks/04_xgboost_timeseries.ipynb` | XGBoost sobre la matriz de variables; CV consciente del tiempo; **multi-paso recursivo vs directo** (y la brecha 1 paso vs 60 pasos); importancia por ganancia; modelo en el **registry**. |
| 5 | `notebooks/05_ensembles.ipynb` | **Ensambles de pronósticos**: por qué combinar funciona (M3/M4, *forecast combination puzzle*); media / mediana / pesos por inverso del RMSE / **stacking** con ventana de validación honesta; comparación completa en MLflow. |
| 6 | `notebooks/06_timeseries_clustering.ipynb` | Perfiles de carga diarios (24 h) del propio dataset; z-normalización; Euclidiana vs **DTW** (recurrencia + camino de deformación); `tslearn` TimeSeriesKMeans (DTW+DBA) y jerárquico de scipy; validación con **ARI**; **NMF** como reducción de dimensión interpretable ($X \approx WH$, patrones horarios arquetípicos) + k-means en el espacio reducido. |

Cada notebook con modelos calcula pronósticos sobre los **últimos 60 días**,
reporta **MSE, RMSE, MAE, MAPE y sMAPE**, grafica el pronóstico contra la
realidad y registra todo (parámetros, métricas, figuras y modelo) en MLflow.

## MLflow: tracking y registry (como en el Módulo 2)

Los notebooks usan los mismos helpers del Módulo 2
(`utils/mlflow_helpers.py`): `setup_mlflow()`, `log_and_register()` y
`register_best_run()`. El backend se elige con `MLFLOW_BACKEND` en un archivo
`.env` (ver `test_env.example`):

- **`dagshub`** — el MLflow gestionado que DagsHub adjunta al repo del curso
  (OAuth del paquete `dagshub` la primera vez).
- **`local`** — un servidor en `http://localhost:5000` (puedes levantar el
  stack de `module2-advanced-ml/mlflow/docker-compose.yml`); si no está
  disponible cae automáticamente a un almacén SQLite local (`./mlruns.db`),
  que también soporta el Model Registry.

Experimentos: `module3-01-decomposition` … `module3-06-clustering`. Modelos
registrados: `module3-power-statistical` (mejor modelo estadístico por sMAPE),
`module3-power-xgboost` y `module3-power-stacking`.

## Chuleta de conceptos clave

**Descomposición**
- Aditivo: `y_t = T_t + S_t + R_t`; multiplicativo: `y_t = T_t · S_t · R_t`
  (→ aditivo en escala log). Nuestra serie es ≈ aditiva.
- STL (LOESS): estacionalidad que evoluciona, suavidad ajustable, `robust=True`.

**Métricas** (sobre el horizonte de prueba)
- MSE / RMSE (cuadráticas), MAE (robusta), MAPE (%, explota si `y≈0`),
  sMAPE (% simétrica, acotada [0, 200]).

**Suavizamiento exponencial**
- SES: `ℓ_t = α·y_t + (1-α)·ℓ_{t-1}` (pronóstico plano).
- Holt: + tendencia `b_t` (mejor amortiguada con φ).
- Holt-Winters: + estacionalidad `s_t` de periodo `m` (aditiva o multiplicativa).

**ARIMA / SARIMA**
- `φ(L)(1-L)^d y_t = c + θ(L) ε_t`; estacional: `Φ(L^m)`, `(1-L^m)^D`, `Θ(L^m)`.
- ACF se corta → MA(q); PACF se corta → AR(p); ambas decaen → ARMA.
- Validar: Ljung-Box (p-valor alto), Q-Q, ACF de residuos; reportar IC.

**Pronóstico con ML**
- Variables: rezagos, móviles (`shift(1)` contra la fuga), calendario,
  Fourier `sin/cos(2πkt/m)`.
- Validar con `TimeSeriesSplit`; multi-paso recursivo (errores se acumulan)
  vs directo (un modelo por horizonte).

**Ensambles de pronósticos**
- Combinar familias diversas cancela errores no correlacionados.
- Media simple ≫ difícil de vencer; pesos `w ∝ 1/RMSE_val`; stacking con
  regresión no negativa sobre una ventana de validación.

**Clustering y reducción de dimensión**
- DTW: `D(i,j) = c(i,j) + min{D(i-1,j), D(i,j-1), D(i-1,j-1)}`.
- z-normaliza → agrupa por forma; valida con el Índice de Rand Ajustado.
- NMF: `X ≈ WH` con `X, W, H ≥ 0` (kW crudos, sin z-norm) → `H` = patrones
  horarios aditivos, `W` = activaciones por día (24 → r dims); elige `r` con
  el codo del error de reconstrucción y agrupa en `W` con k-means euclidiano.

## Prerrequisitos

- Python 3.10–3.12.
- Soltura con numpy / pandas / matplotlib y ML supervisado básico.
- Módulo 1 (feature engineering) y Módulo 2 (ML avanzado, MLflow) vistos.

## Cómo ejecutar

Con [uv](https://docs.astral.sh/uv/) (recomendado, igual que el Módulo 2):

```bash
cd module3-time-series
uv sync
cp test_env.example .env      # y elige MLFLOW_BACKEND=local|dagshub
uv run jupyter lab            # abre notebooks/ y ejecuta de arriba a abajo
```

O con pip clásico:

```bash
cd module3-time-series
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Los notebooks se entregan **sin salida ejecutada** a propósito; corre cada
celda en orden. Se generan de forma programática — no edites a mano los
archivos `.ipynb`. Para regenerarlos:

```bash
python3 _build_notebooks.py
```

## Estructura del módulo

```
module3-time-series/
├── README.md
├── pyproject.toml           # entorno gestionado con uv (uv sync)
├── requirements.txt         # alternativa pip
├── test_env.example         # plantilla del .env (backend de MLflow)
├── _build_notebooks.py      # generador (fuente de verdad de los notebooks)
├── utils/
│   └── mlflow_helpers.py    # setup_mlflow / log_and_register / register_best_run
├── notebooks/
│   ├── 01_decomposition.ipynb
│   ├── 02_arima.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_xgboost_timeseries.ipynb
│   ├── 05_ensembles.ipynb
│   └── 06_timeseries_clustering.ipynb
└── data/                    # cachés del dataset UCI (se descargan solos)
```
