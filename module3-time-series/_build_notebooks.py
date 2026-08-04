"""
Generador de los notebooks del Módulo 3 - Series de Tiempo.

Ejecutar:  python3 _build_notebooks.py

Este script ensambla los seis notebooks dentro de ./notebooks usando nbformat.
NO edites los archivos .ipynb a mano; edita este script y vuelve a ejecutarlo.

Todos los notebooks usan el MISMO dataset real:
"Individual Household Electric Power Consumption" (UCI ML Repository #235)
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
y registran sus experimentos en MLflow (mismos helpers que el Módulo 2).
"""

import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(HERE, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def build(filename, cells):
    nb = new_notebook()
    nb_cells = []
    for kind, src in cells:
        if kind == "md":
            nb_cells.append(new_markdown_cell(src))
        elif kind == "code":
            nb_cells.append(new_code_cell(src))
        else:
            raise ValueError(kind)
    nb["cells"] = nb_cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    }
    path = os.path.join(NB_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("escrito", path)


# ---------------------------------------------------------------------------
# Snippets compartidos por todos los notebooks
# ---------------------------------------------------------------------------

SETUP_CODE = r'''
import os, sys, warnings
warnings.filterwarnings("ignore")

# Hacemos importable utils/ tanto si el notebook corre desde notebooks/ como
# desde la raíz del repositorio.
_here = os.getcwd()
for cand in (os.path.join(_here, "..", "utils"), os.path.join(_here, "utils"),
             os.path.join(_here, "..", "..", "module3-time-series", "utils")):
    cand = os.path.abspath(cand)
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow_helpers import setup_mlflow, log_and_register, register_best_run

plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["axes.grid"] = True
np.random.seed(42)
print("Versión de MLflow:", mlflow.__version__)
'''.strip()


DATA_MD = r"""## El dataset: consumo eléctrico de un hogar (UCI #235)

Usaremos en **todo el módulo** el dataset
[Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
del repositorio de UCI: mediciones eléctricas de una vivienda en Sceaux
(Francia) tomadas **cada minuto** entre diciembre de 2006 y noviembre de 2010
(~2 millones de filas, ~1.25% de minutos faltantes marcados con `?`).

Trabajaremos con la **potencia activa global** (`Global_active_power`, en
kilovatios) agregada a dos resoluciones:

- **`daily`** — promedio diario (≈1,440 días): la serie principal para
  pronóstico.
- **`hourly`** — promedio horario (≈34,500 horas): la usaremos para perfiles
  de carga intradía.

La primera ejecución descarga el zip (~20 MB) y deja un caché CSV en
`../data/`; las siguientes leen el caché. Los huecos (incluidos un par de
cortes de varios días) se rellenan con interpolación temporal. Si no hay
internet, se genera un **respaldo sintético** con estacionalidad semanal y
anual para que el notebook siga corriendo."""


DATA_CODE = r'''
# ---------------------------------------------------------------------------
# Carga del dataset UCI #235 (con caché local y respaldo sintético offline)
# ---------------------------------------------------------------------------
import io, zipfile, urllib.request

UCI_ZIP_URL = ("https://archive.ics.uci.edu/static/public/235/"
               "individual+household+electric+power+consumption.zip")

def _data_dir():
    for cand in ("../data", "data", "module3-time-series/data"):
        cand = os.path.abspath(cand)
        if os.path.isdir(cand):
            return cand
    cand = os.path.abspath("../data")
    os.makedirs(cand, exist_ok=True)
    return cand

DATA_DIR = _data_dir()
DAILY_CSV = os.path.join(DATA_DIR, "household_power_daily.csv")
HOURLY_CSV = os.path.join(DATA_DIR, "household_power_hourly.csv")

def load_household_power():
    """Devuelve (daily, hourly): potencia activa global media, en kW."""
    if os.path.isfile(DAILY_CSV) and os.path.isfile(HOURLY_CSV):
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True).iloc[:, 0]
        hourly = pd.read_csv(HOURLY_CSV, index_col=0, parse_dates=True).iloc[:, 0]
        return daily.asfreq("D"), hourly.asfreq("h")

    print("Descargando el dataset UCI #235 (~20 MB)...")
    raw = urllib.request.urlopen(UCI_ZIP_URL, timeout=180).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open("household_power_consumption.txt") as fh:
            df = pd.read_csv(fh, sep=";", na_values=["?"], low_memory=False,
                             usecols=["Date", "Time", "Global_active_power"])
    ts = pd.to_datetime(df["Date"] + " " + df["Time"],
                        format="%d/%m/%Y %H:%M:%S")
    power = pd.Series(df["Global_active_power"].astype(float).to_numpy(),
                      index=ts, name="global_active_power_kw").sort_index()

    # Agregamos y rellenamos huecos por interpolación temporal (~1.25% de
    # minutos faltantes + un par de cortes de varios días).
    daily = power.resample("D").mean().interpolate(method="time")
    hourly = power.resample("h").mean().interpolate(method="time")
    daily = daily.iloc[1:-1]                       # primer/último día parciales
    hourly = hourly.loc[daily.index.min():
                        daily.index.max() + pd.Timedelta(hours=23)]
    daily.to_frame().to_csv(DAILY_CSV)
    hourly.to_frame().to_csv(HOURLY_CSV)
    return daily.asfreq("D"), hourly.asfreq("h")

try:
    daily, hourly = load_household_power()
    print(f"daily : {daily.index.min().date()} .. {daily.index.max().date()} "
          f"(n={len(daily)})")
    print(f"hourly: n={len(hourly)}")
except Exception as e:
    print("No se pudo descargar el dataset:", repr(e))
    print("Usando RESPALDO SINTÉTICO (estacionalidad semanal + anual).")
    rng = np.random.default_rng(7)
    idx = pd.date_range("2006-12-17", "2010-11-25", freq="D")
    t = np.arange(len(idx))
    daily = pd.Series(
        1.1
        + 0.35 * np.cos(2 * np.pi * (t - 20) / 365.25)   # invierno alto
        + 0.10 * (idx.dayofweek >= 5)                     # fin de semana
        + rng.normal(0, 0.12, len(idx)),
        index=idx, name="global_active_power_kw").clip(lower=0.1).asfreq("D")
    hidx = pd.date_range(idx.min(), idx.max() + pd.Timedelta(hours=23), freq="h")
    hh = hidx.hour.to_numpy()
    base = daily.reindex(pd.DatetimeIndex(hidx.date)).to_numpy()
    profile = 0.6 + 0.35 * np.sin(2 * np.pi * (hh - 14) / 24) \
              + 0.25 * ((hh >= 18) & (hh <= 22))
    hourly = pd.Series(base * profile + rng.normal(0, 0.05, len(hidx)),
                       index=hidx, name=daily.name).clip(lower=0.05).asfreq("h")
'''.strip()


METRICS_MD = r"""## Métricas de evaluación de pronósticos

Estas cinco métricas acompañarán **todos** los modelos del módulo. Con
$e_t = y_t - \hat y_t$ sobre un horizonte de prueba de $H$ pasos:

$$
\text{MSE} = \frac{1}{H}\sum_t e_t^2
\qquad
\text{RMSE} = \sqrt{\text{MSE}}
\qquad
\text{MAE} = \frac{1}{H}\sum_t |e_t|
$$

$$
\text{MAPE} = \frac{100}{H}\sum_t \left|\frac{e_t}{y_t}\right|
\qquad
\text{sMAPE} = \frac{100}{H}\sum_t \frac{2\,|e_t|}{|y_t| + |\hat y_t|}
$$

- **MSE / RMSE** penalizan mucho los errores grandes (cuadráticos); el RMSE
  vuelve a las unidades originales (kW).
- **MAE** es robusta y directamente interpretable ("nos equivocamos en X kW
  en promedio").
- **MAPE** es un porcentaje fácil de comunicar, pero **explota cuando
  $y_t \approx 0$** y penaliza más los sobre-pronósticos que los
  sub-pronósticos (asimetría).
- **sMAPE** corrige parcialmente esa asimetría normalizando por el promedio de
  $|y_t|$ y $|\hat y_t|$; queda acotada en $[0, 200]\%$. Fue la métrica de las
  competencias M3/M4.

> Regla práctica: optimiza/compara con RMSE **y** una métrica porcentual
> (sMAPE), y reporta MAE porque es la más fácil de explicar al negocio."""


METRICS_CODE = r'''
# ---------------------------------------------------------------------------
# Métricas de pronóstico + gráfico estándar — se usan en TODOS los notebooks.
# ---------------------------------------------------------------------------
def forecast_metrics(y_true, y_pred):
    """MSE, RMSE, MAE, MAPE y sMAPE como dict {nombre: float}."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    return {
        "MSE":   mse,
        "RMSE":  float(np.sqrt(mse)),
        "MAE":   float(np.mean(np.abs(err))),
        "MAPE":  float(np.mean(np.abs(err) / np.abs(y_true)) * 100.0),
        "sMAPE": float(np.mean(2.0 * np.abs(err)
                               / (np.abs(y_true) + np.abs(y_pred))) * 100.0),
    }

def print_metrics(name, m):
    print(f"{name:<26s} MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  "
          f"MAE={m['MAE']:.4f}  MAPE={m['MAPE']:.2f}%  sMAPE={m['sMAPE']:.2f}%")

def metrics_table(metrics_by_model):
    """dict {modelo: dict_de_métricas} -> DataFrame ordenado por sMAPE."""
    return (pd.DataFrame(metrics_by_model).T
            .sort_values("sMAPE").round(4))

def plot_forecast(train, test, forecasts, title="", tail=180, ci=None):
    """Cola del train + test real + uno o varios pronósticos.

    forecasts : dict {nombre: pd.Series indexada como test}
    ci        : tupla opcional (lower, upper) para una banda de confianza
    Devuelve la figura (útil para loggearla en MLflow).
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    train.iloc[-tail:].plot(ax=ax, label="train (cola)", color="0.65")
    test.plot(ax=ax, label="real (test)", color="black", lw=2)
    for name, fc in forecasts.items():
        fc.plot(ax=ax, label=name, lw=1.8)
    if ci is not None:
        ax.fill_between(test.index, ci[0], ci[1], alpha=0.2, label="IC 95%")
    ax.set_title(title)
    ax.set_ylabel("potencia activa media (kW)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig
'''.strip()


FEATURES_CODE = r'''
# ---------------------------------------------------------------------------
# Matriz de variables supervisada para pronóstico diario (sin fuga temporal).
# Construida y explicada en el notebook 03 — aquí la reutilizamos tal cual.
# ---------------------------------------------------------------------------
def make_features(s, lags=(1, 2, 3, 7, 14, 28, 365),
                  roll_windows=(7, 28), fourier=((7, 2), (365.25, 3))):
    df = pd.DataFrame({"y": s})
    df["t_index"] = np.arange(len(df))

    # rezagos
    for L in lags:
        df[f"lag_{L}"] = df["y"].shift(L)

    # estadísticos móviles SOLO del pasado: shift(1) antes de rolling
    past = df["y"].shift(1)
    for w in roll_windows:
        df[f"rollmean_{w}"] = past.rolling(w).mean()
        df[f"rollstd_{w}"] = past.rolling(w).std()
        df[f"rollmin_{w}"] = past.rolling(w).min()
        df[f"rollmax_{w}"] = past.rolling(w).max()

    # calendario
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # estacionalidad de Fourier (semanal y anual)
    tt = df["t_index"].to_numpy()
    for period, K in fourier:
        for k in range(1, K + 1):
            df[f"sin_{int(period)}_{k}"] = np.sin(2 * np.pi * k * tt / period)
            df[f"cos_{int(period)}_{k}"] = np.cos(2 * np.pi * k * tt / period)
    return df
'''.strip()


XGB_RECURSIVE_CODE = r'''
def xgb_recursive_forecast(model, history, h):
    """Pronóstico multi-paso RECURSIVO: extiende la serie un día a la vez,
    recalculando las variables (los rezagos recientes van siendo predichos)."""
    hist = history.copy()
    preds = []
    for _ in range(h):
        next_date = hist.index[-1] + pd.Timedelta(days=1)
        f_next = make_features(
            pd.concat([hist, pd.Series([np.nan], index=[next_date])]))
        row = f_next.drop(columns=["y"]).iloc[[-1]]
        yhat = float(model.predict(row)[0])
        hist = pd.concat([hist, pd.Series([yhat], index=[next_date])])
        preds.append(yhat)
    idx = pd.date_range(history.index[-1] + pd.Timedelta(days=1),
                        periods=h, freq="D")
    return pd.Series(preds, index=idx, name="xgb_recursive")
'''.strip()


# ===========================================================================
# 01_decomposition.ipynb
# ===========================================================================
def nb01_decomposition():
    cells = []
    cells.append(("md", r"""# 01 - El dataset, descomposición y estacionariedad

**Módulo 3 - Series de Tiempo | ML Avanzado**

Una *serie de tiempo* es una secuencia de observaciones ordenadas en el tiempo,
$\{y_t\}_{t=1}^{T}$. A diferencia de los datos tabulares i.i.d., **el orden
temporal lleva información**: las observaciones cercanas están correlacionadas,
y esa correlación es justo lo que explotaremos para pronosticar.

En este notebook:

1. Conocemos el **dataset del módulo** (consumo eléctrico de un hogar, UCI #235).
2. Los componentes de una serie; modelos **aditivos vs multiplicativos**.
3. Descomposición clásica (`seasonal_decompose`) y **STL**.
4. **Estacionariedad**: estadísticos móviles, ACF/PACF y prueba ADF.
5. **Pronósticos baseline** + las **métricas** (MSE, RMSE, MAE, MAPE, sMAPE)
   que usaremos en todo el módulo, registradas en **MLflow**."""))

    cells.append(("code", SETUP_CODE))
    cells.append(("md", DATA_MD))
    cells.append(("code", DATA_CODE))

    cells.append(("code", r'''ax = daily.plot(title="Consumo eléctrico del hogar - potencia activa media diaria")
ax.set_ylabel("kW")
plt.tight_layout(); plt.show()

# Dos patrones saltan a la vista:
#  - ESTACIONALIDAD ANUAL: inviernos altos (calefacción), veranos bajos
#    (¡y los huecos de vacaciones de agosto!).
#  - Mucho ruido de día a día: el comportamiento humano es irregular.'''))

    cells.append(("md", r"""### Estacionalidad a varias escalas

El consumo de un hogar tiene estructura en **tres** escalas temporales:
anual (clima), semanal (rutinas laborales vs fin de semana) e intradía
(mañana/noche). Veámoslas por separado."""))

    cells.append(("code", r'''fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))

# anual: promedio por mes
daily.groupby(daily.index.month).mean().plot(ax=axes[0], marker="o")
axes[0].set_title("Promedio por mes (estacionalidad anual)")
axes[0].set_xlabel("mes"); axes[0].set_ylabel("kW")

# semanal: promedio por día de la semana
daily.groupby(daily.index.dayofweek).mean().plot(ax=axes[1], marker="o")
axes[1].set_title("Promedio por día de la semana")
axes[1].set_xlabel("0=lun ... 6=dom")

# intradía: promedio por hora (serie horaria)
hourly.groupby(hourly.index.hour).mean().plot(ax=axes[2], marker="o")
axes[2].set_title("Promedio por hora del día")
axes[2].set_xlabel("hora")

plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""## 1. Los componentes de una serie de tiempo

Un modelo conceptual muy usado descompone la serie observada en cuatro partes:

$$
y_t = T_t + S_t + C_t + R_t
$$

- **Tendencia $T_t$** — la dirección de largo plazo.
- **Estacionalidad $S_t$** — un patrón que se repite con periodo *fijo y
  conocido* $m$ (aquí: $m=7$ días para la semana, $m\approx365$ para el año).
- **Ciclo $C_t$** — fluctuaciones largas **sin periodo fijo** (ciclos
  económicos); en la práctica se absorbe en la tendencia.
- **Residual $R_t$** — lo que queda; idealmente ruido blanco.

### ¿Aditivo o multiplicativo?

**Aditivo** — la amplitud estacional es constante sin importar el nivel:
$y_t = T_t + S_t + R_t$.

**Multiplicativo** — la amplitud **escala con el nivel** (picos cada vez más
grandes): $y_t = T_t \times S_t \times R_t$. Truco clave: en escala
**logarítmica** se vuelve aditivo, porque
$\log y_t = \log T_t + \log S_t + \log R_t$.

En nuestro dataset la amplitud del ciclo anual es razonablemente estable (no
crece con los años — el hogar no consume cada vez más), así que trabajaremos
con el modelo **aditivo**. Compara esto con la clásica serie *AirPassengers*,
el ejemplo canónico multiplicativo."""))

    cells.append(("md", r"""## 2. Descomposición clásica (`seasonal_decompose`)

`statsmodels` implementa la descomposición clásica basada en promedios
móviles: (1) estima $\hat T_t$ con un promedio móvil centrado de longitud $m$;
(2) quita la tendencia; (3) estima $\hat S_t$ promediando por posición dentro
del periodo; (4) el resto es $\hat R_t$.

Para verla con claridad suavizamos primero a **promedio semanal** y
descomponemos el ciclo anual ($m = 52$ semanas)."""))

    cells.append(("code", r'''from statsmodels.tsa.seasonal import seasonal_decompose

weekly = daily.resample("W").mean()
dec = seasonal_decompose(weekly, model="additive", period=52)
fig = dec.plot()
fig.set_size_inches(12, 8)
fig.suptitle("Descomposición clásica aditiva (serie semanal, periodo=52)", y=1.02)
plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""### Limitaciones de la descomposición clásica

- La tendencia queda **indefinida en los bordes** (se pierde media ventana en
  cada extremo — mira los NaN al inicio/fin del panel *Trend*).
- El componente estacional está forzado a ser **idéntico cada periodo**.
- **No es robusta** a atípicos (un corte de luz contamina la tendencia).

## 3. Descomposición STL

**STL** (*Seasonal-Trend decomposition using LOESS*, Cleveland et al. 1990)
ajusta suavizadores LOESS en un bucle interno/externo. Siempre es **aditiva**
(para efectos multiplicativos: aplicar sobre $\log y_t$). Ventajas:

- El componente estacional **puede evolucionar** en el tiempo (`seasonal`).
- Suavidad de la tendencia **ajustable** (`trend`).
- **Robusta** a atípicos con `robust=True` (pesos que amortiguan residuos
  grandes — ideal con los huecos/vacaciones de este hogar).

Descomponemos la serie **diaria** con periodo semanal $m=7$: la "tendencia"
de STL absorberá el ciclo anual (que para $m=7$ es de baja frecuencia)."""))

    cells.append(("code", r'''from statsmodels.tsa.seasonal import STL

stl = STL(daily, period=7, robust=True)
res = stl.fit()
fig = res.plot()
fig.set_size_inches(12, 8)
fig.suptitle("STL de la serie diaria (period=7, robust=True)", y=1.02)
plt.tight_layout(); plt.show()

print("Desviación estándar del residual:", round(res.resid.std(), 4))'''))

    cells.append(("code", r'''# El componente estacional semanal de STL puede derivar con el tiempo:
seasonal = res.seasonal
fig, ax = plt.subplots(figsize=(10, 4))
for year in [2007, 2008, 2009, 2010]:
    s = seasonal[seasonal.index.year == year]
    ax.plot(s.groupby(s.index.dayofweek).mean().values, marker="o", label=str(year))
ax.set_xticks(range(7), ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"])
ax.set_ylabel("efecto estacional (kW)")
ax.set_title("Patrón semanal medio estimado por STL, año a año")
ax.legend(); plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""## 4. Estacionariedad

Una serie es **(débilmente) estacionaria** si sus propiedades estadísticas no
cambian con el tiempo:

$$
\mathbb{E}[y_t] = \mu, \qquad
\operatorname{Var}(y_t) = \sigma^2, \qquad
\operatorname{Cov}(y_t, y_{t+k}) = \gamma(k) \ \text{depende solo de } k .
$$

¿Por qué importa? La teoría clásica (AR, MA, ARMA — notebook 02) **supone
estacionariedad**. Tendencia y estacionalidad la violan; las quitamos con
**diferenciación** — con el operador de rezago $L y_t = y_{t-1}$: primera
diferencia $(1-L)$, diferencia estacional $(1-L^m)$ — modelamos la versión
estacionaria e invertimos al final.

Diagnóstico en tres frentes: estadísticos móviles, ACF/PACF y la prueba ADF."""))

    cells.append(("code", r'''def plot_rolling(series, window, title=""):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series, color="0.7", label="serie")
    ax.plot(series.rolling(window).mean(), color="C0", label=f"media móvil ({window})")
    ax.plot(series.rolling(window).std(), color="C3", label=f"desv. móvil ({window})")
    ax.set_title(title); ax.legend()
    plt.tight_layout(); plt.show()

plot_rolling(daily, 30, "Serie diaria - la media móvil oscila con las estaciones")

# diferencia estacional semanal + primera diferencia
stationary = daily.diff(1).diff(7).dropna()   # (1-L)(1-L^7) y_t
plot_rolling(stationary, 30, "(1-L)(1-L^7) y_t - media y varianza ~constantes")'''))

    cells.append(("md", r"""### ACF y PACF

- La **ACF** $\rho(k) = \gamma(k)/\gamma(0)$ mide la correlación entre $y_t$ y
  $y_{t-k}$, *incluyendo* los efectos indirectos vía rezagos intermedios.
- La **PACF** mide esa correlación **tras eliminar** la influencia de los
  rezagos $1..k-1$.

Señal delatora: en una serie no estacionaria la ACF decae *muy lento*. En el
notebook 02 usaremos estas gráficas para elegir los órdenes de un ARIMA."""))

    cells.append(("code", r'''from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
plot_acf(daily, lags=60, ax=axes[0, 0])
axes[0, 0].set_title("ACF - serie diaria (picos cada 7 días + decae lento)")
plot_pacf(daily, lags=60, ax=axes[0, 1], method="ywm")
axes[0, 1].set_title("PACF - serie diaria")
plot_acf(stationary, lags=60, ax=axes[1, 0])
axes[1, 0].set_title("ACF - diferenciada (decae rápido)")
plot_pacf(stationary, lags=60, ax=axes[1, 1], method="ywm")
axes[1, 1].set_title("PACF - diferenciada")
plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""### Prueba de Dickey-Fuller Aumentada (ADF)

Contrasta la existencia de una **raíz unitaria** con la regresión

$$
\Delta y_t = \alpha + \beta t + \gamma\, y_{t-1}
           + \sum_{i=1}^{p} \delta_i\, \Delta y_{t-i} + \varepsilon_t .
$$

- $H_0$: $\gamma = 0$ → raíz unitaria → **no estacionaria**.
- $H_1$: $\gamma < 0$ → **estacionaria**.

**p-valor < 0.05** ⇒ rechazamos $H_0$ ⇒ estacionaria. (Complemento: la prueba
KPSS invierte las hipótesis; usar ambas es buena práctica.)"""))

    cells.append(("code", r'''from statsmodels.tsa.stattools import adfuller

def adf_report(series, name):
    stat, p, lags, *_ = adfuller(series.dropna(), autolag="AIC")
    verdict = "ESTACIONARIA" if p < 0.05 else "NO estacionaria"
    print(f"ADF {name:<38s} stat={stat:8.4f}  p={p:.4f}  -> {verdict}")

adf_report(daily, "serie diaria")
adf_report(daily.diff(7).dropna(), "diferencia estacional (1-L^7)")
adf_report(stationary, "(1-L)(1-L^7)")

# La serie diaria puede pasar el ADF (no tiene tendencia de largo plazo),
# pero la estacionalidad semanal/anual sigue ahí: ADF NO detecta
# estacionalidad, solo raíces unitarias. Por eso miramos también la ACF.'''))

    cells.append(("md", METRICS_MD))
    cells.append(("code", METRICS_CODE))

    cells.append(("md", r"""## 5. Pronósticos *baseline* — el punto de referencia obligatorio

Antes de cualquier modelo sofisticado necesitamos **baselines**: si tu SARIMA
o tu XGBoost no le ganan a estos, algo anda mal. Reservamos los **últimos 60
días** como prueba (la división siempre es **cronológica**) y evaluamos:

- **Naive**: $\hat y_{T+h} = y_T$ — repetir el último valor.
- **Naive estacional**: $\hat y_{T+h} = y_{T+h-7}$ — repetir la última semana.
- **Media móvil**: $\hat y_{T+h} = \tfrac{1}{7}\sum_{i=0}^{6} y_{T-i}$."""))

    cells.append(("code", r'''H = 60
train, test = daily.iloc[:-H], daily.iloc[-H:]
print(f"train: {train.index.min().date()} .. {train.index.max().date()} (n={len(train)})")
print(f"test : {test.index.min().date()} .. {test.index.max().date()} (n={len(test)})")

baselines = {
    "naive": pd.Series(train.iloc[-1], index=test.index),
    "naive_estacional": pd.Series(
        np.tile(train.iloc[-7:].to_numpy(), H // 7 + 1)[:H], index=test.index),
    "media_movil_7": pd.Series(train.iloc[-7:].mean(), index=test.index),
}

all_metrics = {}
for name, fc in baselines.items():
    all_metrics[name] = forecast_metrics(test, fc)
    print_metrics(name, all_metrics[name])

metrics_table(all_metrics)'''))

    cells.append(("code", r'''fig = plot_forecast(train, test, baselines,
                    title="Baselines vs realidad (últimos 60 días)")'''))

    cells.append(("md", r"""## 6. Registro en MLflow

Igual que en el Módulo 2, cada modelo (aquí, cada baseline) se registra como
un **run** de MLflow con sus parámetros, sus 5 métricas y la figura del
pronóstico. El backend (`local` con Docker/SQLite o `dagshub` en la nube) se
elige con `MLFLOW_BACKEND` en el archivo `.env` — ver `test_env.example`."""))

    cells.append(("code", r'''setup_mlflow("module3-01-decomposition", backend="dagshub")

for name, fc in baselines.items():
    log_and_register(
        run_name=f"baseline-{name}",
        params={"model": name, "horizon_days": H, "dataset": "uci-household-power"},
        metrics=all_metrics[name],
        tags={"notebook": "01_decomposition", "familia": "baseline"},
        figures={"plots/forecast.png": fig},
    )
print("Baselines registrados. Revisa la UI de MLflow para compararlos.")'''))

    cells.append(("md", r"""## Resumen

- El dataset del módulo: **consumo eléctrico de un hogar** (UCI #235), con
  estacionalidad **anual, semanal e intradía** — trabajamos con la serie
  diaria (modelo **aditivo**: la amplitud no escala con el nivel).
- `seasonal_decompose` es simple pero rígida; **STL** permite estacionalidad
  que evoluciona y es **robusta** a atípicos.
- **Estacionariedad** = media/varianza/autocovarianza constantes; se
  diagnostica con estadísticos móviles + ACF/PACF + **ADF**, y se consigue
  diferenciando: $(1-L)$, $(1-L^7)$.
- Definimos las métricas del módulo (**MSE, RMSE, MAE, MAPE, sMAPE**) y
  fijamos los **baselines** a vencer, registrados en **MLflow**.

Siguiente: **02 — Suavizamiento exponencial y ARIMA/SARIMA**."""))

    build("01_decomposition.ipynb", cells)


# ===========================================================================
# 02_arima.ipynb  (Exponential Smoothing + ARIMA/SARIMA)
# ===========================================================================
def nb02_arima():
    cells = []
    cells.append(("md", r"""# 02 - Suavizamiento Exponencial y ARIMA / SARIMA

**Módulo 3 - Series de Tiempo | ML Avanzado**

Los dos pilares del pronóstico estadístico clásico:

1. **Suavizamiento exponencial** (SES → Holt → Holt-Winters): promedios
   ponderados del pasado con pesos que decaen exponencialmente; descomponen la
   serie en nivel / tendencia / estacionalidad y los actualizan de forma
   recursiva.
2. **ARIMA / SARIMA**: modelan la serie (diferenciada hasta ser estacionaria)
   como función lineal de sus valores pasados y de sus errores pasados.

Ambas familias se evalúan con las métricas del módulo sobre el **mismo
horizonte de 60 días**, y todos los modelos quedan **registrados en MLflow**
(tracking + Model Registry)."""))

    cells.append(("code", SETUP_CODE))
    cells.append(("code", DATA_CODE))
    cells.append(("code", METRICS_CODE))

    cells.append(("code", r'''H = 60
train, test = daily.iloc[:-H], daily.iloc[-H:]
print(f"train: {train.index.min().date()} .. {train.index.max().date()} (n={len(train)})")
print(f"test : {test.index.min().date()} .. {test.index.max().date()} (n={len(test)})")

forecasts, all_metrics, fitted = {}, {}, {}   # acumularemos todos los modelos aquí'''))

    cells.append(("md", r"""## 1. Suavizamiento Exponencial Simple (SES)

Idea: el pronóstico es un **promedio ponderado de todo el pasado**, con pesos
que decaen exponencialmente — lo reciente pesa más. Con nivel $\ell_t$ y
parámetro de suavizamiento $\alpha \in (0, 1]$:

$$
\ell_t = \alpha\, y_t + (1-\alpha)\, \ell_{t-1},
\qquad
\hat y_{T+h|T} = \ell_T \quad \forall h .
$$

Desenrollando la recursión: $\ell_t = \alpha \sum_{j=0}^{\infty} (1-\alpha)^j
y_{t-j}$ — de ahí el nombre *exponencial*.

- $\alpha \to 1$: solo importa el último valor (≈ naive).
- $\alpha \to 0$: promedio de largo plazo, muy suave.
- El pronóstico es **plano**: SES no modela tendencia ni estacionalidad.

`statsmodels` estima $\alpha$ (y el nivel inicial) maximizando la
verosimilitud."""))

    cells.append(("code", r'''from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing, Holt, ExponentialSmoothing)

ses = SimpleExpSmoothing(train, initialization_method="estimated").fit()
fc = ses.forecast(H); fc.index = test.index
forecasts["SES"], fitted["SES"] = fc, ses
all_metrics["SES"] = forecast_metrics(test, fc)

print("alpha óptimo:", round(float(ses.params["smoothing_level"]), 4))
print_metrics("SES", all_metrics["SES"])'''))

    cells.append(("md", r"""## 2. Holt: agregando tendencia (y amortiguándola)

Holt añade una componente de **tendencia** $b_t$ con su propio suavizamiento
$\beta$:

$$
\ell_t = \alpha\, y_t + (1-\alpha)(\ell_{t-1} + b_{t-1})
\qquad
b_t = \beta\, (\ell_t - \ell_{t-1}) + (1-\beta)\, b_{t-1}
$$

$$
\hat y_{T+h|T} = \ell_T + h\, b_T .
$$

Extrapolar una recta indefinidamente suele ser demasiado optimista; la
variante **amortiguada** (*damped*, Gardner) multiplica la tendencia por
$\phi \in (0,1)$: $\hat y_{T+h|T} = \ell_T + (\phi + \phi^2 + \dots + \phi^h)
b_T$, que converge a una asíntota. En la práctica el *damped trend* es uno de
los pronosticadores univariados más difíciles de vencer."""))

    cells.append(("code", r'''holt = Holt(train, damped_trend=True, initialization_method="estimated").fit()
fc = holt.forecast(H); fc.index = test.index
forecasts["Holt"], fitted["Holt"] = fc, holt
all_metrics["Holt"] = forecast_metrics(test, fc)

print({k: round(float(v), 4) for k, v in holt.params.items()
       if k in ("smoothing_level", "smoothing_trend", "damping_trend")})
print_metrics("Holt (damped)", all_metrics["Holt"])'''))

    cells.append(("md", r"""## 3. Holt-Winters: agregando estacionalidad

La versión completa añade un componente **estacional** $s_t$ de periodo $m$
(aquí $m=7$, la semana) con suavizamiento $\gamma$. En la forma **aditiva**:

$$
\ell_t = \alpha\,(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})
$$
$$
b_t = \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\, b_{t-1}
\qquad
s_t = \gamma\,(y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)\, s_{t-m}
$$
$$
\hat y_{T+h|T} = \ell_T + h\, b_T + s_{T+h-m\lceil h/m \rceil} .
$$

La forma **multiplicativa** reemplaza restas por divisiones
($y_t / s_{t-m}$, etc.) y se usa cuando la amplitud estacional escala con el
nivel. Nuestra serie es aproximadamente aditiva (notebook 01), así que usamos
`seasonal="add"` con tendencia amortiguada."""))

    cells.append(("code", r'''hw = ExponentialSmoothing(
    train, trend="add", damped_trend=True,
    seasonal="add", seasonal_periods=7,
    initialization_method="estimated").fit()
fc = hw.forecast(H); fc.index = test.index
forecasts["Holt-Winters"], fitted["Holt-Winters"] = fc, hw
all_metrics["Holt-Winters"] = forecast_metrics(test, fc)

print({k: round(float(v), 4) for k, v in hw.params.items()
       if k in ("smoothing_level", "smoothing_trend",
                "smoothing_seasonal", "damping_trend")})
print_metrics("Holt-Winters", all_metrics["Holt-Winters"])'''))

    cells.append(("code", r'''fig_es = plot_forecast(
    train, test,
    {k: forecasts[k] for k in ("SES", "Holt", "Holt-Winters")},
    title="Familia de suavizamiento exponencial (test = últimos 60 días)")
# SES y Holt son (casi) planos; Holt-Winters recupera el patrón semanal.'''))

    cells.append(("md", r"""## 4. ARIMA: los bloques de construcción

Recordatorio del operador de rezago: $L\, y_t = y_{t-1}$, $L^k y_t = y_{t-k}$.

**AR(p)** — el presente como combinación lineal del pasado reciente:
$$
y_t = c + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \varepsilon_t
\quad\Longleftrightarrow\quad
\phi(L)\, y_t = c + \varepsilon_t .
$$

**MA(q)** — el presente como combinación de los *shocks* recientes:
$$
y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}
= c + \theta(L)\, \varepsilon_t .
$$

**ARMA(p, q)** — ambos: $\phi(L)\, y_t = c + \theta(L)\, \varepsilon_t$.
Supone **estacionariedad**.

**ARIMA(p, d, q)** — la "I" de *integrada*: diferenciamos $d$ veces primero:
$$
\phi(L)\,(1-L)^d\, y_t = c + \theta(L)\, \varepsilon_t .
$$

**SARIMA$(p,d,q)(P,D,Q)_m$** — polinomios estacionales en $L^m$ y diferencia
estacional $(1-L^m)^D$:
$$
\phi(L)\,\Phi(L^m)\,(1-L)^d (1-L^m)^D\, y_t
= c + \theta(L)\,\Theta(L^m)\,\varepsilon_t .
$$

### Identificación con ACF/PACF (serie ya estacionaria)

| Patrón | Sugiere |
|---|---|
| ACF se corta tras el rezago $q$; PACF decae | **MA(q)** |
| PACF se corta tras el rezago $p$; ACF decae | **AR(p)** |
| Ambas decaen gradualmente | **ARMA(p, q)** |
| ACF decae *muy* lento | falta diferenciar ($d$+1) |
| Picos en los rezagos $m, 2m, \dots$ | términos estacionales $P/Q$ |"""))

    cells.append(("code", r'''from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# ¿Cuánta diferenciación necesitamos?
print(f"ADF serie original      : p = {adfuller(train, autolag='AIC')[1]:.4f}")
diffed = train.diff(7).dropna()           # d=0, D=1 (m=7)
print(f"ADF tras (1-L^7)        : p = {adfuller(diffed, autolag='AIC')[1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
plot_acf(diffed, lags=42, ax=axes[0]); axes[0].set_title("ACF de (1-L^7) y_t")
plot_pacf(diffed, lags=42, ax=axes[1], method="ywm"); axes[1].set_title("PACF")
plt.tight_layout(); plt.show()

# Lectura: el ADF de la serie original ya rechaza raíz unitaria (no hay
# tendencia de largo plazo) -> d=0. La estacionalidad semanal pide D=1 con
# m=7. En la serie diferenciada: pico negativo en el rezago 7 de la ACF
# (MA estacional Q=1) y autocorrelación de corto plazo (p=1, q=1).
# Candidato: SARIMA(1,0,1)(0,1,1)_7.'''))

    cells.append(("md", r"""## 5. Ajuste del SARIMA y diagnóstico de residuos

Ajustamos SARIMA$(1,0,1)(0,1,1)_7$. Si el modelo capturó la estructura, los
residuos deben ser **ruido blanco**:

- **Ljung-Box**: $H_0$ = residuos independientes hasta el rezago $h$;
  *queremos* p-valores **altos** (no rechazar).
- **Q-Q plot**: puntos sobre la recta → residuos ~normales.
- **ACF de residuos**: dentro de la banda de confianza."""))

    cells.append(("code", r'''from statsmodels.tsa.statespace.sarimax import SARIMAX

ORDER, SORDER = (1, 0, 1), (0, 1, 1, 7)
sarima = SARIMAX(train, order=ORDER, seasonal_order=SORDER,
                 enforce_stationarity=False, enforce_invertibility=False
                 ).fit(disp=False)
print(sarima.summary().tables[1])'''))

    cells.append(("code", r'''fig = sarima.plot_diagnostics(figsize=(12, 8))
plt.tight_layout(); plt.show()

from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(sarima.resid, lags=[7, 14, 28], return_df=True))
print("\nQueremos lb_pvalue GRANDES -> residuos ~ ruido blanco.")'''))

    cells.append(("md", r"""## 6. Pronóstico SARIMA con intervalos de confianza

`get_forecast` devuelve la media y un **intervalo de confianza** por paso; el
intervalo se ensancha con el horizonte porque la incertidumbre se acumula."""))

    cells.append(("code", r'''fc_obj = sarima.get_forecast(steps=H)
fc = fc_obj.predicted_mean; fc.index = test.index
ci = fc_obj.conf_int(alpha=0.05); ci.index = test.index

forecasts["SARIMA"], fitted["SARIMA"] = fc, sarima
all_metrics["SARIMA"] = forecast_metrics(test, fc)
print_metrics("SARIMA(1,0,1)(0,1,1)7", all_metrics["SARIMA"])

fig_sarima = plot_forecast(
    train, test, {"SARIMA": fc},
    title="SARIMA(1,0,1)(0,1,1)$_7$ con IC 95%",
    ci=(ci.iloc[:, 0], ci.iloc[:, 1]))'''))

    cells.append(("md", r"""## 7. `auto_arima` (pmdarima) — selección automática (opcional)

`pmdarima.auto_arima` busca los órdenes automáticamente: $d$/$D$ con pruebas
de raíz unitaria (ADF/KPSS, OCSB/CH) y $p,q,P,Q$ minimizando AIC/BIC con una
búsqueda *stepwise*. Trátalo con sentido crítico: corre siempre los mismos
diagnósticos de residuos. (Celda opcional — `pip install pmdarima`.)"""))

    cells.append(("code", r'''try:
    import pmdarima as pm
    auto = pm.auto_arima(train, seasonal=True, m=7, stepwise=True,
                         suppress_warnings=True, error_action="ignore",
                         max_p=3, max_q=3, max_P=2, max_Q=2)
    print("auto_arima eligió:", auto.order, auto.seasonal_order,
          "| AIC:", round(auto.aic(), 1))
except Exception as e:
    print("pmdarima no disponible (opcional):", repr(e))
    print("Seguimos con el SARIMA(1,0,1)(0,1,1)_7 identificado a mano.")'''))

    cells.append(("md", r"""## 8. Comparación de todos los modelos"""))

    cells.append(("code", r'''table = metrics_table(all_metrics)
display(table)

ax = table["sMAPE"].plot(kind="barh", figsize=(8, 3.5), color="C0")
ax.set_xlabel("sMAPE (%)  (menor = mejor)")
ax.set_title("Suavizamiento exponencial vs SARIMA - test de 60 días")
plt.tight_layout(); plt.show()

fig_all = plot_forecast(train, test, forecasts,
                        title="Todos los modelos estadísticos vs realidad")'''))

    cells.append(("md", r"""## 9. Tracking y Registry en MLflow

Un **run por modelo** con: hiperparámetros, las 5 métricas, la figura del
pronóstico comparativo y el propio modelo serializado (flavor
`mlflow.statsmodels`). Al final promovemos el mejor (mínimo sMAPE) al **Model
Registry** con `register_best_run` — igual que en el Módulo 2."""))

    cells.append(("code", r'''setup_mlflow("module3-02-statistical-models", backend="dagshub")

def es_params(fit_result):
    keys = ("smoothing_level", "smoothing_trend",
            "smoothing_seasonal", "damping_trend")
    out = {}
    for k in keys:
        v = fit_result.params.get(k, np.nan)
        if v == v:                       # descarta NaN
            out[k] = round(float(v), 4)
    return out

run_params = {
    "SES":          {"model": "SimpleExpSmoothing", **es_params(fitted["SES"])},
    "Holt":         {"model": "Holt", "damped": True, **es_params(fitted["Holt"])},
    "Holt-Winters": {"model": "ExponentialSmoothing", "trend": "add",
                     "damped": True, "seasonal": "add", "m": 7,
                     **es_params(fitted["Holt-Winters"])},
    "SARIMA":       {"model": "SARIMAX", "order": str(ORDER),
                     "seasonal_order": str(SORDER)},
}

for name in forecasts:
    log_and_register(
        run_name=name,
        params={**run_params[name], "horizon_days": H,
                "dataset": "uci-household-power"},
        metrics=all_metrics[name],
        model=fitted[name],
        flavor="statsmodels",
        tags={"notebook": "02_arima", "familia": "estadistica"},
        figures={"plots/comparacion.png": fig_all},
    )

register_best_run("module3-02-statistical-models", metric="sMAPE",
                  registered_model_name="module3-power-statistical",
                  mode="min")'''))

    cells.append(("md", r"""## 10. Serving: consumir el modelo desde el Registry

Cerramos el **ciclo de gestión del modelo**: entrenar → trackear → registrar
→ **servir**. Un proceso consumidor (una API, un job batch de pronóstico) no
reentrena nada ni conoce este notebook: solo necesita el **nombre** del modelo
en el registry y pide la última versión con la URI

```
models:/module3-power-statistical/latest
```

Usamos el flavor **nativo** (`mlflow.statsmodels.load_model`) porque devuelve
el objeto de resultados de statsmodels con su API completa
(`.forecast(steps)`, intervalos de confianza...); el wrapper genérico
`pyfunc` espera un DataFrame de entrada y no encaja con la firma
`predict(start, end)` de los modelos estadísticos."""))

    cells.append(("code", r'''MODEL_NAME = "module3-power-statistical"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

client = mlflow.MlflowClient()
versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
latest = max(int(v.version) for v in versions)
print(f"Registry: '{MODEL_NAME}' tiene {len(versions)} versión(es); "
      f"sirviendo la v{latest}")

serving_model = mlflow.statsmodels.load_model(MODEL_URI)
print("Modelo cargado:", type(serving_model.model).__name__)

# El modelo registrado se ajustó sobre `train`: su forecast arranca justo
# donde termina el entrenamiento, es decir, sobre la ventana de test.
fc_serving = serving_model.forecast(steps=H)
fc_serving.index = test.index

m_serving = forecast_metrics(test, fc_serving)
print_metrics("modelo servido (registry)", m_serving)

plot_forecast(train, test, {"pronóstico servido": fc_serving},
              title=f"Serving desde el registry: {MODEL_NAME} v{latest}");'''))

    cells.append(("md", r"""En producción el patrón es el mismo pero con dos diferencias:

1. **Reajuste con datos frescos**: antes de pronosticar el futuro real, el
   modelo se reentrena (o se actualiza con `.append()` en statsmodels) con
   todas las observaciones disponibles — aquí mantuvimos el ajuste original
   para poder comparar contra el test.
2. **Versiones explícitas**: en vez de `latest`, un servicio serio fija la
   versión (`models:/nombre/3`) o usa un *alias* (`@champion`) que se mueve
   solo tras validar la nueva versión."""))

    cells.append(("md", r"""## Resumen

- **SES** suaviza solo el nivel (pronóstico plano); **Holt** añade tendencia
  (mejor amortiguada con $\phi$); **Holt-Winters** añade estacionalidad
  aditiva o multiplicativa — aquí $m=7$.
- **ARIMA**: AR usa valores pasados, MA usa shocks pasados, la "I" diferencia
  hasta la estacionariedad; **SARIMA** añade $(P,D,Q)_m$. Órdenes: $d, D$ por
  pruebas ADF + inspección; $p, q, P, Q$ por ACF/PACF (o `auto_arima`).
- Valida SIEMPRE los residuos (Ljung-Box, Q-Q, ACF) y reporta **intervalos de
  confianza**.
- Todos los modelos quedaron en **MLflow** y el mejor (por sMAPE) en el
  **Model Registry** — y cerramos el ciclo **consumiéndolo** desde
  `models:/module3-power-statistical/latest` para pronosticar.

Siguiente: **03 — Ingeniería de variables para series de tiempo**, el puente
hacia el pronóstico con machine learning."""))

    build("02_arima.ipynb", cells)


# ===========================================================================
# 03_feature_engineering.ipynb
# ===========================================================================
def nb03_features():
    cells = []
    cells.append(("md", r"""# 03 - Ingeniería de Variables para Series de Tiempo

**Módulo 3 - Series de Tiempo | ML Avanzado**

Los modelos de ML (XGBoost, redes, regresión lineal...) no entienden de
"series": entienden de **tablas**. El puente es reformular el pronóstico como
**regresión supervisada**:

$$
y_t = f\big(\underbrace{y_{t-1}, y_{t-7}, \dots}_{\text{rezagos}},\;
\underbrace{\bar y_{t-7:t-1}, \dots}_{\text{ventanas móviles}},\;
\underbrace{\text{día, mes, Fourier}}_{\text{calendario}}\big) + \varepsilon_t
$$

Este notebook construye esa matriz de variables **sin fuga temporal** y la
valida con un modelo lineal. La regla de oro:

> **Toda variable de la fila $t$ debe poder calcularse usando solo información
> disponible antes de $t$** (o información de calendario, que se conoce por
> adelantado)."""))

    cells.append(("code", SETUP_CODE))
    cells.append(("code", DATA_CODE))
    cells.append(("code", METRICS_CODE))

    cells.append(("md", r"""## 1. Rezagos (*lags*)

El rezago $k$ es simplemente $y_{t-k}$. ¿Cuáles incluir? Los que la propia
serie sugiera: la **ACF** nos dice qué rezagos correlacionan con el presente.
Para una serie diaria con semana y año: $1, 2, 3$ (persistencia de corto
plazo), $7, 14, 28$ (estacionalidad semanal) y $365$ (estacionalidad anual)."""))

    cells.append(("code", r'''lags_df = pd.DataFrame({
    "y": daily,
    "lag_1": daily.shift(1),
    "lag_7": daily.shift(7),
    "lag_365": daily.shift(365),
}).dropna()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, lag in zip(axes, ["lag_1", "lag_7", "lag_365"]):
    ax.scatter(lags_df[lag], lags_df["y"], s=4, alpha=0.35)
    r = lags_df["y"].corr(lags_df[lag])
    ax.set_title(f"y vs {lag}   (r = {r:.2f})")
    ax.set_xlabel(lag); ax.set_ylabel("y")
plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""## 2. Estadísticos de ventana móvil

Media, desviación, mínimo y máximo sobre una ventana **del pasado**:

$$
\text{rollmean}_w(t) = \frac{1}{w} \sum_{i=1}^{w} y_{t-i}
$$

Codifican el **nivel local** y la **volatilidad local** suavizando el ruido.

⚠️ **La trampa de fuga más común**: `y.rolling(7).mean()` en la fila $t$
**incluye $y_t$** — ¡la variable contiene la respuesta! La solución es
desplazar primero: `y.shift(1).rolling(7).mean()`."""))

    cells.append(("code", r'''# MAL  (fuga): la ventana de la fila t incluye y_t
leaky = daily.rolling(7).mean()
# BIEN (sin fuga): primero shift(1), la ventana termina en t-1
safe = daily.shift(1).rolling(7).mean()

comp = pd.DataFrame({"y": daily, "rolling_LEAKY": leaky,
                     "rolling_SAFE": safe}).iloc[400:460]
ax = comp.plot(figsize=(12, 4), style=["k-", "C3--", "C0-"])
ax.set_title("La versión con fuga 'copia' la serie; la segura va un paso atrás")
plt.tight_layout(); plt.show()

print("corr(y, leaky) =", round(daily.corr(leaky), 4),
      " | corr(y, safe) =", round(daily.corr(safe), 4))'''))

    cells.append(("md", r"""## 3. Variables de calendario

El calendario se conoce **por adelantado**, así que no hay riesgo de fuga:
día de la semana, mes, fin de semana, festivos... En este hogar el efecto
fin de semana es visible (más tiempo en casa)."""))

    cells.append(("code", r'''cal = pd.DataFrame({"y": daily,
                    "dayofweek": daily.index.dayofweek,
                    "month": daily.index.month})

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
cal.boxplot(column="y", by="dayofweek", ax=axes[0])
axes[0].set_title("Consumo por día de la semana (0=lun)"); axes[0].set_xlabel("")
cal.boxplot(column="y", by="month", ax=axes[1])
axes[1].set_title("Consumo por mes"); axes[1].set_xlabel("")
fig.suptitle("")
plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""## 4. Estacionalidad de Fourier

Los enteros de calendario (`month = 1..12`) imponen un **orden falso**
(diciembre y enero quedan "lejos" siendo vecinos). Para efectos periódicos
suaves usamos **términos de Fourier** de periodo $m$:

$$
\sin\!\Big(\frac{2\pi k t}{m}\Big), \quad
\cos\!\Big(\frac{2\pi k t}{m}\Big), \qquad k = 1, \dots, K .
$$

Unos pocos armónicos $K$ aproximan cualquier forma estacional suave, y además
le dan a los modelos de árboles una representación **continua** del ciclo
anual. Demostración: regresión lineal solo con Fourier anual ($K=3$)."""))

    cells.append(("code", r'''from sklearn.linear_model import LinearRegression

t_idx = np.arange(len(daily))
Xf = np.column_stack(
    [np.sin(2 * np.pi * k * t_idx / 365.25) for k in (1, 2, 3)] +
    [np.cos(2 * np.pi * k * t_idx / 365.25) for k in (1, 2, 3)])
smooth = LinearRegression().fit(Xf, daily.values).predict(Xf)

fig, ax = plt.subplots(figsize=(12, 4))
daily.plot(ax=ax, color="0.75", label="serie diaria")
ax.plot(daily.index, smooth, color="C1", lw=2.5,
        label="Fourier anual (K=3) ajustado por regresión lineal")
ax.set_title("6 columnas sin/cos capturan el ciclo anual completo")
ax.legend(); plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""## 5. La matriz de variables completa

Juntamos todo en una función. **Esta misma función se reutiliza en los
notebooks 04 (XGBoost) y 05 (ensambles)** — en producción viviría en un
*feature store* (Módulo 1)."""))

    cells.append(("code", FEATURES_CODE))

    cells.append(("code", r'''feat = make_features(daily).dropna()
X, y = feat.drop(columns=["y"]), feat["y"]
print("matriz de variables:", X.shape, "| primeras columnas:", list(X.columns[:8]))
feat.head(3)'''))

    cells.append(("md", r"""## 6. Validación consciente del tiempo

**¿Por qué no k-fold aleatorio?** Mezclaría filas del futuro en el
entrenamiento (los rezagos del fold de validación aparecen como objetivo en el
de entrenamiento y viceversa) → puntajes optimistas que se desploman en
producción. En series de tiempo **la validación siempre va después del
entrenamiento**:

- **Ventana expansiva** — `TimeSeriesSplit`: el train crece, se valida el
  siguiente tramo.
- **Ventana deslizante** — train de tamaño fijo que avanza."""))

    cells.append(("code", r'''from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=60)
fig, ax = plt.subplots(figsize=(11, 3))
for i, (tr, va) in enumerate(tscv.split(X)):
    ax.plot(tr, [i] * len(tr), ".", color="C0", markersize=2)
    ax.plot(va, [i] * len(va), ".", color="C3", markersize=3)
ax.set_yticks(range(5)); ax.set_ylabel("fold"); ax.set_xlabel("fila (tiempo)")
ax.set_title("TimeSeriesSplit: azul = train (expansivo), rojo = validación")
plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""## 7. Sanity check: ¿las variables tienen señal?

Un modelo **lineal regularizado (Ridge)** sobre la matriz de variables. Si
esta base tan simple ya vence a los baselines del notebook 01, la ingeniería
de variables está funcionando. Evaluamos a **1 paso** sobre los últimos 60
días (predicción del día $t$ usando los valores *reales* hasta $t-1$; el
multi-paso "honesto" llega en el notebook 04)."""))

    cells.append(("code", r'''from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H = 60
X_train, X_test = X.iloc[:-H], X.iloc[-H:]
y_train, y_test = y.iloc[:-H], y.iloc[-H:]

ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))

cv_rmse = []
for tr, va in tscv.split(X_train):
    ridge.fit(X_train.iloc[tr], y_train.iloc[tr])
    pred = ridge.predict(X_train.iloc[va])
    cv_rmse.append(float(np.sqrt(np.mean((y_train.iloc[va] - pred) ** 2))))
print("RMSE por fold:", [round(s, 4) for s in cv_rmse],
      "| promedio:", round(np.mean(cv_rmse), 4))

ridge.fit(X_train, y_train)
pred_test = pd.Series(ridge.predict(X_test), index=y_test.index)

ridge_metrics = forecast_metrics(y_test, pred_test)
print_metrics("Ridge (1 paso)", ridge_metrics)

fig_ridge = plot_forecast(y_train, y_test, {"Ridge (1 paso)": pred_test},
                          title="Regresión Ridge sobre la matriz de variables")'''))

    cells.append(("code", r'''# ¿Qué variables pesan más? (coeficientes sobre variables estandarizadas)
coefs = pd.Series(ridge.named_steps["ridge"].coef_, index=X.columns)
ax = coefs.reindex(coefs.abs().sort_values().index).tail(15).plot(
    kind="barh", figsize=(8, 5))
ax.set_title("Ridge: 15 coeficientes de mayor magnitud")
plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""## 8. Registro en MLflow y exportación de la matriz"""))

    cells.append(("code", r'''setup_mlflow("module3-03-feature-engineering", backend="dagshub")

FEATURES_CSV = os.path.join(DATA_DIR, "household_features_daily.csv")
feat.to_csv(FEATURES_CSV)
print("matriz exportada a", FEATURES_CSV)

log_and_register(
    run_name="ridge-feature-sanity-check",
    params={"model": "Ridge", "alpha": 10.0, "n_features": X.shape[1],
            "eval": "one-step-ahead", "horizon_days": H,
            "dataset": "uci-household-power"},
    metrics={**ridge_metrics, "cv_rmse_mean": float(np.mean(cv_rmse))},
    model=ridge,
    flavor="sklearn",
    input_example=X_test.head(3),
    tags={"notebook": "03_feature_engineering"},
    figures={"plots/forecast.png": fig_ridge},
    artifact_files={FEATURES_CSV: "features"},
)'''))

    cells.append(("md", r"""## Resumen

- El pronóstico con ML = **regresión supervisada** sobre una matriz de
  variables construida solo con el pasado + calendario.
- Tres familias: **rezagos** (guiados por la ACF), **ventanas móviles** (con
  `shift(1)` — la trampa de fuga nº 1) y **calendario/Fourier** (los términos
  $\sin/\cos$ evitan el orden falso de los enteros y capturan ciclos suaves).
- Valida con **`TimeSeriesSplit`**, nunca con k-fold aleatorio.
- Un Ridge sobre estas variables ya compite — la matriz queda exportada y la
  función `make_features` se reutiliza en los notebooks 04 y 05.

Siguiente: **04 — XGBoost para pronóstico** (multi-paso recursivo, importancia
de variables, registry)."""))

    build("03_feature_engineering.ipynb", cells)


# ===========================================================================
# 04_xgboost_timeseries.ipynb
# ===========================================================================
def nb04_xgboost():
    cells = []
    cells.append(("md", r"""# 04 - XGBoost para Pronóstico de Series de Tiempo

**Módulo 3 - Series de Tiempo | ML Avanzado**

Los ensambles de árboles como XGBoost no son "modelos de series de tiempo",
pero están entre los mejores pronosticadores en la práctica *una vez que el
problema se reformula como regresión supervisada* (notebook 03). Dos detalles
importan:

- Los árboles **no extrapolan**: una hoja devuelve una constante, así que no
  pueden seguir una tendencia fuera del rango visto. Nuestra serie no tiene
  tendencia de largo plazo, y además los rezagos/índice de tiempo transportan
  el nivel.
- La validación debe ser **consciente del tiempo** (`TimeSeriesSplit`), y el
  pronóstico multi-paso necesita una **estrategia** (recursiva vs directa)."""))

    cells.append(("code", SETUP_CODE))
    cells.append(("code", DATA_CODE))
    cells.append(("code", METRICS_CODE))
    cells.append(("code", FEATURES_CODE))

    cells.append(("code", r'''feat = make_features(daily).dropna()
X, y = feat.drop(columns=["y"]), feat["y"]

H = 60
train_s, test_s = daily.iloc[:-H], daily.iloc[-H:]
print("matriz:", X.shape, "| test:", test_s.index.min().date(),
      "..", test_s.index.max().date())'''))

    cells.append(("md", r"""## 1. Validación cruzada consciente del tiempo"""))

    cells.append(("code", r'''from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

XGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, n_jobs=-1)

# CV solo sobre la parte de entrenamiento (el test de 60 días queda intacto)
mask_train = feat.index <= train_s.index.max()
X_train, y_train = X[mask_train], y[mask_train]

tscv = TimeSeriesSplit(n_splits=5, test_size=60)
cv_scores = []
for tr, va in tscv.split(X_train):
    m = XGBRegressor(**XGB_PARAMS)
    m.fit(X_train.iloc[tr], y_train.iloc[tr])
    pred = m.predict(X_train.iloc[va])
    cv_scores.append(forecast_metrics(y_train.iloc[va], pred))

cv_df = pd.DataFrame(cv_scores).round(4)
cv_df.index.name = "fold"
display(cv_df)
print("RMSE promedio CV:", round(cv_df["RMSE"].mean(), 4))'''))

    cells.append(("md", r"""## 2. Multi-paso: recursivo vs directo

Para un horizonte $H$:

**Recursivo** — *un* modelo de 1 paso; sus predicciones se reinyectan como
rezagos para avanzar: $\hat y_{t+1} \to$ rezago $\to \hat y_{t+2} \to \dots$
*Pro*: un solo modelo, usa todos los datos. *Contra*: los errores **se
acumulan**.

**Directo** — un modelo *por horizonte* $h$, cada uno predice $y_{t+h}$ desde
variables conocidas en $t$. *Pro*: sin acumulación. *Contra*: $H$ modelos y
pronósticos que pueden quedar irregulares entre horizontes.

Implementamos la **recursiva** (la más común) para comparar de igual a igual
con SARIMA/Holt-Winters, que también pronostican 60 días "a ciegas"."""))

    cells.append(("code", r'''final = XGBRegressor(**XGB_PARAMS).fit(X_train, y_train)'''))

    cells.append(("code", XGB_RECURSIVE_CODE))

    cells.append(("code", r'''fc_rec = xgb_recursive_forecast(final, train_s, H)
fc_rec.index = test_s.index

xgb_metrics = forecast_metrics(test_s, fc_rec)
print_metrics("XGBoost recursivo", xgb_metrics)

fig_xgb = plot_forecast(train_s, test_s, {"XGBoost recursivo": fc_rec},
                        title="XGBoost - pronóstico multi-paso recursivo (60 días)")'''))

    cells.append(("md", r"""### Referencia: evaluación a 1 paso

La brecha entre la métrica a 1 paso (rezagos reales) y la multi-paso
(rezagos predichos) mide cuánto **se acumulan** los errores recursivos."""))

    cells.append(("code", r'''mask_test = feat.index > train_s.index.max()
pred_1step = pd.Series(final.predict(X[mask_test]), index=y[mask_test].index)
one_step_metrics = forecast_metrics(y[mask_test], pred_1step)

comp = metrics_table({"XGB 1 paso": one_step_metrics,
                      "XGB recursivo (60 pasos)": xgb_metrics})
display(comp)'''))

    cells.append(("md", r"""## 3. Importancia de variables

`importance_type="gain"`: mejora media del objetivo cuando la variable se usa
en un corte. (Para atribución con signo y por-predicción: valores SHAP.)"""))

    cells.append(("code", r'''imp = pd.Series(
    final.get_booster().get_score(importance_type="gain")).sort_values()
ax = imp.tail(15).plot(kind="barh", figsize=(8, 6))
ax.set_title("XGBoost - importancia por ganancia (top 15)")
plt.tight_layout(); plt.show()
# Es de esperar que dominen lag_1/lag_7, las medias móviles y Fourier anual.'''))

    cells.append(("md", r"""## 4. Tracking + Model Registry en MLflow

Registramos el run con CV + test + figura + el modelo (flavor `xgboost`), y lo
publicamos en el registry como `module3-power-xgboost`."""))

    cells.append(("code", r'''setup_mlflow("module3-04-xgboost", backend="dagshub")

log_and_register(
    run_name="xgboost-recursive-h60",
    params={**XGB_PARAMS, "strategy": "recursive", "horizon_days": H,
            "n_features": X.shape[1], "dataset": "uci-household-power"},
    metrics={**xgb_metrics,
             "cv_rmse_mean": float(cv_df["RMSE"].mean()),
             "one_step_RMSE": one_step_metrics["RMSE"]},
    model=final,
    flavor="xgboost",
    registered_model_name="module3-power-xgboost",
    input_example=X_train.head(3),
    tags={"notebook": "04_xgboost", "familia": "ml"},
    figures={"plots/forecast.png": fig_xgb},
)

# Alternativa: promover después el mejor run del experimento
# register_best_run("module3-04-xgboost", metric="sMAPE",
#                   registered_model_name="module3-power-xgboost", mode="min")'''))

    cells.append(("md", r"""## 5. Serving: consumir el modelo desde el Registry

Cerramos el **ciclo de gestión del modelo**: entrenar → trackear → registrar
→ **servir**. Cargamos la última versión registrada con el wrapper genérico
**`pyfunc`** — el contrato universal de serving de MLflow: no importa el
flavor con que se guardó (xgboost, sklearn, pytorch...), el consumidor
siempre ve lo mismo, `.predict(DataFrame) -> array`. Es exactamente lo que
expone `mlflow models serve` detrás de un endpoint REST.

Como el contrato es el mismo, nuestro `xgb_recursive_forecast` funciona
**sin cambios** con el modelo servido: le da igual recibir el `XGBRegressor`
en memoria o el `pyfunc` que vino del registry."""))

    cells.append(("code", r'''MODEL_NAME = "module3-power-xgboost"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

serving_model = mlflow.pyfunc.load_model(MODEL_URI)
print("Firma del modelo (validada en cada predict):")
print(serving_model.metadata.signature)

# 1) Reproducimos el pronóstico del test con el modelo SERVIDO:
fc_serving = xgb_recursive_forecast(serving_model, train_s, H)
fc_serving.index = test_s.index
print_metrics("XGBoost servido (registry)", forecast_metrics(test_s, fc_serving))
print("¿Idéntico al modelo en memoria?",
      bool(np.allclose(fc_serving.to_numpy(), fc_rec.to_numpy())))'''))

    cells.append(("code", r'''# 2) Pronóstico REAL a futuro: en producción alimentamos TODA la serie
#    disponible y proyectamos más allá del último dato observado.
H_FUT = 30
fc_future = xgb_recursive_forecast(serving_model, daily, H_FUT)

fig, ax = plt.subplots(figsize=(12, 4))
daily.iloc[-120:].plot(ax=ax, label="observado (últimos 120 días)")
fc_future.plot(ax=ax, color="C3", lw=2, label=f"pronóstico a {H_FUT} días")
ax.axvline(daily.index[-1], color="0.5", ls="--", lw=1)
ax.set_ylabel("kW"); ax.legend()
ax.set_title(f"Serving: {MODEL_NAME}/latest pronosticando el futuro real")
plt.tight_layout(); plt.show()

# En producción este bloque ES el job batch de pronóstico: carga por nombre
# desde el registry + datos frescos -> pronóstico. Nada del entrenamiento
# viaja al consumidor; solo el nombre del modelo y el feature pipeline
# (make_features), que por eso debe versionarse junto al modelo.'''))

    cells.append(("md", r"""## Resumen

- XGBoost pronostica vía la matriz de variables del notebook 03; validación
  con **`TimeSeriesSplit`**, nunca aleatoria.
- **Recursivo** = 1 modelo, errores que se acumulan (medimos la brecha 1 paso
  vs 60 pasos); **directo** = $H$ modelos sin acumulación.
- La **importancia por ganancia** confirma qué variables aportan (rezagos
  recientes, medias móviles, Fourier anual).
- Modelo versionado en el **Model Registry** (`module3-power-xgboost`) y
  **consumido de vuelta** vía `pyfunc` para pronosticar el futuro real —
  ciclo completo: entrenar → registrar → servir.

Siguiente: **05 — Ensambles de pronósticos**: combinar SARIMA, Holt-Winters y
XGBoost suele ganarle a cada uno por separado."""))

    build("04_xgboost_timeseries.ipynb", cells)


# ===========================================================================
# 05_ensembles.ipynb
# ===========================================================================
def nb05_ensembles():
    cells = []
    cells.append(("md", r"""# 05 - Ensambles de Pronósticos

**Módulo 3 - Series de Tiempo | ML Avanzado**

Combinar pronósticos es una de las ideas más antiguas y robustas del área
(Bates & Granger, 1969). La evidencia empírica es contundente: en las
competencias **M3/M4**, las combinaciones simples vencieron sistemáticamente a
casi todos los modelos individuales.

**¿Por qué funciona?** Cada modelo comete errores distintos: SARIMA captura la
autocorrelación lineal, Holt-Winters el nivel/estacionalidad suave, XGBoost
las no-linealidades del calendario. Si los errores no están perfectamente
correlacionados, promediar **cancela parte del error** — el mismo argumento
varianza-reducción del bagging (Módulo 2), aplicado a pronósticos.

La paradoja conocida como *forecast combination puzzle*: la **media simple**
es dificilísima de vencer con esquemas de pesos "óptimos", porque los pesos
estimados añaden su propia varianza.

Probaremos cuatro combinaciones:

1. **Media simple** — $\hat y = \frac{1}{M}\sum_m \hat y^{(m)}$
2. **Mediana** — robusta a un modelo que se descarrile.
3. **Pesos por inverso del error** — $w_m \propto 1 / \text{RMSE}^{(m)}_{val}$
4. **Stacking** — un meta-modelo lineal aprende los pesos sobre una ventana de
   validación (el análogo temporal del stacking del Módulo 2)."""))

    cells.append(("code", SETUP_CODE))
    cells.append(("code", DATA_CODE))
    cells.append(("code", METRICS_CODE))
    cells.append(("code", FEATURES_CODE))

    cells.append(("md", r"""## 1. Tres ventanas: train / validación / test

Para aprender pesos sin hacer trampa necesitamos una ventana de **validación**
separada del test final:

```
|--------------- train ---------------|-- val (60d) --|-- test (60d) --|
```

- Los modelos base se ajustan en *train* y pronostican *val* → con esos
  errores estimamos los **pesos**.
- Luego se reajustan en *train+val* y pronostican *test* → ahí comparamos
  **todo** (bases y ensambles) de forma honesta."""))

    cells.append(("code", r'''H = 60
trainval, test = daily.iloc[:-H], daily.iloc[-H:]
train, val = trainval.iloc[:-H], trainval.iloc[-H:]
print(f"train: n={len(train)} | val: n={len(val)} | test: n={len(test)}")'''))

    cells.append(("md", r"""## 2. Los modelos base

Cuatro pronosticadores de familias distintas (la **diversidad** es el
ingrediente clave de un ensamble): naive estacional, Holt-Winters, SARIMA y
XGBoost recursivo — los mismos de los notebooks 01, 02 y 04."""))

    cells.append(("code", XGB_RECURSIVE_CODE))

    cells.append(("code", r'''from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

XGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, n_jobs=-1)

def fc_snaive(history, h, idx):
    vals = np.tile(history.iloc[-7:].to_numpy(), h // 7 + 1)[:h]
    return pd.Series(vals, index=idx)

def fc_holtwinters(history, h, idx):
    m = ExponentialSmoothing(history, trend="add", damped_trend=True,
                             seasonal="add", seasonal_periods=7,
                             initialization_method="estimated").fit()
    f = m.forecast(h); f.index = idx
    return f

def fc_sarima(history, h, idx):
    m = SARIMAX(history, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False)
    f = m.forecast(h); f.index = idx
    return f

def fc_xgboost(history, h, idx):
    feat = make_features(history).dropna()
    model = XGBRegressor(**XGB_PARAMS).fit(feat.drop(columns=["y"]), feat["y"])
    f = xgb_recursive_forecast(model, history, h); f.index = idx
    return f, model

BASE = {"snaive": fc_snaive, "holt_winters": fc_holtwinters,
        "sarima": fc_sarima, "xgboost": fc_xgboost}'''))

    cells.append(("code", r'''# Pronósticos sobre VALIDACIÓN (ajuste en train)...
F_val = {}
for name, fn in BASE.items():
    out = fn(train, H, val.index)
    F_val[name] = out[0] if isinstance(out, tuple) else out
F_val = pd.DataFrame(F_val)

# ...y sobre TEST (reajuste en train+val)
F_test, stack_xgb_model = {}, None
for name, fn in BASE.items():
    out = fn(trainval, H, test.index)
    if isinstance(out, tuple):
        F_test[name], stack_xgb_model = out
    else:
        F_test[name] = out
F_test = pd.DataFrame(F_test)

print("errores de VALIDACIÓN (sirven para calcular pesos):")
val_rmse = {}
for name in F_val:
    m = forecast_metrics(val, F_val[name])
    val_rmse[name] = m["RMSE"]
    print_metrics(name, m)'''))

    cells.append(("md", r"""## 3. Las combinaciones

**Pesos por inverso del RMSE de validación** (normalizados a que sumen 1):

$$
w_m = \frac{1 / \text{RMSE}_m}{\sum_j 1 / \text{RMSE}_j}
\qquad
\hat y^{comb}_t = \sum_m w_m\, \hat y^{(m)}_t .
$$

**Stacking**: regresión lineal **sin intercepto y con pesos no negativos**
(para que sea una combinación convexa interpretable) de $y_{val}$ sobre la
matriz de pronósticos de validación."""))

    cells.append(("code", r'''from sklearn.linear_model import LinearRegression

ens = {}
ens["media_simple"] = F_test.mean(axis=1)
ens["mediana"] = F_test.median(axis=1)

w = pd.Series({m: 1.0 / r for m, r in val_rmse.items()})
w /= w.sum()
ens["pesos_inv_rmse"] = (F_test * w).sum(axis=1)
print("pesos inverso-RMSE:", w.round(3).to_dict())

stacker = LinearRegression(positive=True, fit_intercept=False)
stacker.fit(F_val, val)
coef = pd.Series(stacker.coef_, index=F_val.columns)
ens["stacking"] = pd.Series(stacker.predict(F_test), index=test.index)
print("coeficientes stacking:", coef.round(3).to_dict())

# Nota: el stacking puede concentrar todo el peso en 1-2 modelos; los pesos
# inverso-RMSE reparten de forma más conservadora. Compararlos es el punto.'''))

    cells.append(("md", r"""## 4. Evaluación final sobre el test"""))

    cells.append(("code", r'''all_metrics = {}
for name in F_test:
    all_metrics[f"base: {name}"] = forecast_metrics(test, F_test[name])
for name, fc in ens.items():
    all_metrics[f"ens: {name}"] = forecast_metrics(test, fc)

table = metrics_table(all_metrics)
display(table)

colors = ["C1" if i.startswith("ens") else "C0" for i in table.index]
ax = table["sMAPE"].plot(kind="barh", figsize=(9, 4.5), color=colors)
ax.set_xlabel("sMAPE (%)  (menor = mejor)")
ax.set_title("Modelos base (azul) vs ensambles (naranja) - test 60 días")
plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""### Lectura de los resultados

En esta serie **XGBoost domina** con claridad a las demás familias. Cuando eso
pasa, las combinaciones de peso uniforme (media, mediana) **diluyen** al mejor
modelo, y el ensamble más competitivo es el **stacking**, que aprendió en
validación a concentrar el peso en el dominante (míralo en sus coeficientes).

La lección del *forecast combination puzzle* aplica cuando los modelos base
son **comparablemente buenos y diversos** — ahí la media simple brilla. La
moraleja práctica es otra: el ensamble es un **seguro contra apostarle al
modelo equivocado** (ex ante no sabes cuál base ganará en producción), no una
garantía de mejora sobre el mejor modelo visto en retrospectiva."""))

    cells.append(("code", r'''fig_ens = plot_forecast(
    trainval, test,
    {"mejor base": F_test[min(val_rmse, key=val_rmse.get)],
     "media_simple": ens["media_simple"],
     "stacking": ens["stacking"]},
    title="Ensambles de pronósticos vs el mejor modelo base")'''))

    cells.append(("md", r"""## 5. MLflow: un run por método + registry

Registramos bases y ensambles en el experimento `module3-05-ensembles`. El
meta-modelo de stacking (el único ensamble con un objeto entrenado) se publica
en el registry como `module3-power-stacking`."""))

    cells.append(("code", r'''setup_mlflow("module3-05-ensembles", backend="dagshub")

for name in F_test:
    log_and_register(
        run_name=f"base-{name}",
        params={"tipo": "base", "modelo": name, "horizon_days": H,
                "dataset": "uci-household-power"},
        metrics=all_metrics[f"base: {name}"],
        tags={"notebook": "05_ensembles", "familia": "base"},
    )

ens_params = {
    "media_simple":   {"combinacion": "mean"},
    "mediana":        {"combinacion": "median"},
    "pesos_inv_rmse": {"combinacion": "inv_rmse",
                       **{f"w_{k}": round(float(v), 4) for k, v in w.items()}},
    "stacking":       {"combinacion": "stacking_lr_positive",
                       **{f"coef_{k}": round(float(v), 4)
                          for k, v in coef.items()}},
}
for name in ens:
    log_and_register(
        run_name=f"ensemble-{name}",
        params={"tipo": "ensemble", **ens_params[name], "horizon_days": H,
                "dataset": "uci-household-power"},
        metrics=all_metrics[f"ens: {name}"],
        model=stacker if name == "stacking" else None,
        flavor="sklearn",
        registered_model_name=("module3-power-stacking"
                               if name == "stacking" else None),
        input_example=F_val.head(3) if name == "stacking" else None,
        tags={"notebook": "05_ensembles", "familia": "ensemble"},
        figures={"plots/comparacion.png": fig_ens},
    )'''))

    cells.append(("md", r"""## 6. Serving: consumir el stacker desde el Registry

Cerramos el **ciclo de gestión del modelo** consumiendo el meta-modelo desde
el registry con el wrapper genérico **`pyfunc`** (`.predict(DataFrame)`, el
mismo contrato que expone `mlflow models serve`).

Servir un ensamble deja una lección extra: el meta-modelo **no basta**. Sus
*features* son los pronósticos de los modelos base, así que el pipeline de
serving completo es

1. reajustar/pronosticar los **modelos base** sobre los datos frescos
   (aquí ese rol lo cumple `F_test`), y
2. pasar esa matriz al **stacker** cargado del registry.

Si los base cambian (versión, orden de columnas), el meta-modelo servido se
rompe en silencio — por eso la **firma** registrada valida columnas y tipos
en cada `predict`."""))

    cells.append(("code", r'''MODEL_NAME = "module3-power-stacking"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

serving_stacker = mlflow.pyfunc.load_model(MODEL_URI)
print("Firma del modelo (las columnas son los pronósticos base):")
print(serving_stacker.metadata.signature)

# Paso 1 del pipeline de serving: pronósticos base (ya calculados en F_test).
# Paso 2: el meta-modelo del registry los combina.
fc_serving = pd.Series(
    np.asarray(serving_stacker.predict(F_test)).ravel(), index=test.index)

print_metrics("stacking servido (registry)",
              forecast_metrics(test, fc_serving))
print("¿Idéntico al stacker en memoria?",
      bool(np.allclose(fc_serving.to_numpy(), ens["stacking"].to_numpy())))

fig, ax = plt.subplots(figsize=(12, 4))
test.plot(ax=ax, color="black", lw=2, label="realidad (test)")
fc_serving.plot(ax=ax, color="C3", lw=2,
                label="stacking servido desde el registry")
ax.set_ylabel("kW"); ax.legend()
ax.set_title(f"Serving: {MODEL_NAME}/latest combinando los pronósticos base")
plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""## Resumen

- **Combinar pronósticos** de familias diversas cancela errores no
  correlacionados — el hallazgo más repetido de las competencias M3/M4.
- La **media simple** es un punto de partida durísimo de vencer (*forecast
  combination puzzle*): los pesos estimados añaden varianza.
- Alternativas: **mediana** (robusta), **pesos por inverso del RMSE de
  validación** (conservadora) y **stacking** con regresión no negativa
  (agresiva; necesita ventana de validación honesta).
- Si un modelo base **domina** (aquí XGBoost), los pesos uniformes lo diluyen
  y el stacking es el que mejor lo recupera. El ensamble es un *seguro*, no
  una garantía: repórtalo siempre junto a los modelos base.
- Extensiones que valen la pena: modelos **híbridos** (SARIMA + ML sobre sus
  residuos) y re-estimar pesos con ventana rodante.
- Todo quedó versionado en MLflow; el stacker vive en el **registry** y lo
  consumimos de vuelta (`models:/module3-power-stacking/latest`) — pero servir
  un ensamble exige servir también su pipeline de pronósticos base.

Siguiente: **06 — Clustering de perfiles de carga con DTW**."""))

    build("05_ensembles.ipynb", cells)


# ===========================================================================
# 06_timeseries_clustering.ipynb
# ===========================================================================
def nb06_clustering():
    cells = []
    cells.append(("md", r"""# 06 - Clustering de Perfiles de Carga (DTW)

**Módulo 3 - Series de Tiempo | ML Avanzado**

Cerramos el módulo con aprendizaje **no supervisado**: agrupar *series
completas* por la **forma** de su trayectoria. Con nuestro dataset horario,
cada **día** del hogar es una curva de 24 puntos — su *perfil de carga*.
¿Existen arquetipos (día laboral, fin de semana, vacaciones)? Es exactamente
lo que hacen las eléctricas para segmentar clientes.

La herramienta central: **clustering con DTW** (*Dynamic Time Warping*) —
agrupar días por la *forma* de su curva, tolerando rutinas desfasadas en el
tiempo.

Al ser no supervisado, aquí no hay pronóstico que evaluar: en lugar de las
métricas de error usamos el **Índice de Rand Ajustado** contra etiquetas
conocidas (¿laboral o fin de semana?)."""))

    cells.append(("code", SETUP_CODE))
    cells.append(("code", DATA_CODE))

    cells.append(("md", r"""## 1. De la serie horaria a una matriz de perfiles diarios

Pivotamos: filas = días, columnas = las 24 horas. Para que el costo $O(n^2)$
de DTW sea manejable, muestreamos 200 días al azar."""))

    cells.append(("code", r'''prof = hourly.to_frame("kw")
prof["date"] = prof.index.normalize()
prof["hour"] = prof.index.hour
mat = prof.pivot_table(index="date", columns="hour", values="kw").dropna()
print("matriz de perfiles:", mat.shape)

rng = np.random.default_rng(0)
sample_days = np.sort(rng.choice(len(mat), size=min(200, len(mat)),
                                 replace=False))
mat_s = mat.iloc[sample_days]
is_weekend = (pd.DatetimeIndex(mat_s.index).dayofweek >= 5).astype(int)
print(f"muestra: {len(mat_s)} días ({is_weekend.sum()} de fin de semana)")

fig, ax = plt.subplots(figsize=(11, 4))
for row in mat_s.to_numpy()[:60]:
    ax.plot(row, color="0.6", alpha=0.35)
ax.plot(mat_s.mean(axis=0), color="C3", lw=2.5, label="perfil medio")
ax.set_xlabel("hora del día"); ax.set_ylabel("kW")
ax.set_title("Perfiles de carga diarios (60 días de muestra)")
ax.legend(); plt.tight_layout(); plt.show()'''))

    cells.append(("md", r"""## 2. z-normalización: agrupar por forma, no por nivel

Estandarizamos cada perfil a media 0 y varianza 1:

$$
z_t = \frac{y_t - \mu}{\sigma} .
$$

Sin esto, la distancia queda dominada por el **nivel** (invierno vs verano) y
no por el **patrón horario** (madrugar vs trasnochar). Sáltatela solo si el
nivel absoluto es justamente lo que quieres agrupar."""))

    cells.append(("code", r'''def znorm(arr):
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd

dataz = znorm(mat_s.to_numpy())'''))

    cells.append(("md", r"""## 3. ¿Por qué una distancia especial? Euclidiana vs DTW

La distancia **Euclidiana** compara punto a punto:
$d_{euc}(a,b) = \sqrt{\sum_i (a_i - b_i)^2}$ — es **rígida en el tiempo**:
dos días con la misma rutina pero desplazada una hora (¡cenar a las 20 vs a
las 21!) parecen muy distintos.

**Dynamic Time Warping (DTW)** busca el mejor **alineamiento no lineal**
entre los dos ejes de tiempo. Con costo local $c(i,j) = (a_i - b_j)^2$ y la
recurrencia de programación dinámica:

$$
D(i,j) = c(i,j) + \min\{\, D(i-1,j),\; D(i,j-1),\; D(i-1,j-1) \,\},
$$

la distancia es $\sqrt{D(n,p)}$. El **camino de deformación** que traza los
$\min$ debe ser de frontera (de $(1,1)$ a $(n,p)$), monótono y continuo.
Restricciones tipo **banda de Sakoe-Chiba** limitan cuánto se aleja de la
diagonal (acelera y evita deformaciones patológicas). Costo: $O(np)$ por
par."""))

    cells.append(("code", r'''# DTW diminuto y sin dependencias, para que la idea sea concreta.
def dtw_distance(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n, p = len(a), len(b)
    D = np.full((n + 1, p + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, p + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, p])

# Demo con perfiles reales: un día vs el mismo perfil desplazado 2 horas
a = dataz[0]
b = np.roll(a, 2)
print("Euclidiana:", round(float(np.sqrt(np.sum((a - b) ** 2))), 3))
print("DTW       :", round(float(dtw_distance(a, b)), 3),
      " (menor -> reconoce la misma rutina desfasada)")'''))

    cells.append(("md", r"""## 4a. `tslearn`: TimeSeriesKMeans con DTW + centroides DBA

k-means bajo DTW con centroides **DBA** (*DTW Barycenter Averaging*): una
"forma promedio" coherente con DTW, en vez de una media punto a punto.
Empezamos con $k=2$ — la hipótesis natural: ¿laboral vs fin de semana?
(Si `tslearn` no está instalado, seguimos con la vía scipy de la sección
4b.)"""))

    cells.append(("code", r'''km_labels = None
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset

    K = 2
    km = TimeSeriesKMeans(n_clusters=K, metric="dtw", max_iter=10,
                          random_state=0)
    km_labels = km.fit_predict(to_time_series_dataset(dataz))

    fig, axes = plt.subplots(1, K, figsize=(12, 3.6), sharey=True)
    for c, ax in enumerate(np.atleast_1d(axes)):
        for row in dataz[km_labels == c]:
            ax.plot(row, color="0.7", alpha=0.35)
        ax.plot(km.cluster_centers_[c].ravel(), color="C3", lw=2.5,
                label="centroide DBA")
        ax.set_title(f"cluster {c} (n={int(np.sum(km_labels == c))})")
        ax.set_xlabel("hora"); ax.legend()
    fig_clusters = fig
    plt.tight_layout(); plt.show()
except Exception as e:
    print("tslearn no disponible - usaremos el jerárquico de scipy:", repr(e))'''))

    cells.append(("md", r"""## 4b. Jerárquico de scipy sobre la matriz DTW

Alternativa con pocas dependencias: matriz completa de distancias DTW por
pares + clustering **aglomerativo** con enlace promedio. Ventaja: el
**dendrograma** deja elegir $k$ *después* de ver la estructura."""))

    cells.append(("code", r'''from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

N = dataz.shape[0]
dist = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        d = dtw_distance(dataz[i], dataz[j])
        dist[i, j] = dist[j, i] = d
print("matriz DTW:", dist.shape)

Z = linkage(squareform(dist, checks=False), method="average")

fig, ax = plt.subplots(figsize=(12, 4))
dendrogram(Z, ax=ax, no_labels=True,
           color_threshold=0.7 * np.max(Z[:, 2]))
ax.set_title("Dendrograma - jerárquico (enlace promedio) sobre DTW")
ax.set_ylabel("distancia de enlace")
plt.tight_layout(); plt.show()

hier_labels = fcluster(Z, t=2, criterion="maxclust")
print("tamaños de clusters:", np.bincount(hier_labels)[1:])'''))

    cells.append(("md", r"""## 5. Interpretación y validación con ARI

Los IDs de cluster son arbitrarios; el **Índice de Rand Ajustado** (1.0 =
partición idéntica, ≈0 = azar) es invariante al reetiquetado. Contrastamos
contra la etiqueta *fin de semana* y cruzamos con la tabla de contingencia:
¿los clusters descubren la rutina laboral?"""))

    cells.append(("code", r'''from sklearn.metrics import adjusted_rand_score

ari_km = (adjusted_rand_score(is_weekend, km_labels)
          if km_labels is not None else float("nan"))
ari_h = adjusted_rand_score(is_weekend, hier_labels)
if km_labels is not None:
    print("ARI k-means DTW vs fin de semana :", round(ari_km, 3))
print("ARI jerárquico vs fin de semana  :", round(ari_h, 3))

labels_show = km_labels if km_labels is not None else hier_labels
ct = pd.crosstab(pd.Series(labels_show, name="cluster"),
                 pd.Series(np.where(is_weekend == 1, "finde", "laboral"),
                           name="tipo de día"))
display(ct)

# Un ARI cercano a 0 dice que la partición NO coincide con laboral/finde:
# tras z-normalizar y permitir deformación temporal (DTW), la FORMA del día
# laboral y la del finde de este hogar son muy parecidas — la deformación
# absorbe justo los desfases (levantarse más tarde) que los distinguían.
# Lo que sí emerge es un grupo pequeño de días atípicos (vacaciones /
# ausencias, perfiles planos). El ARI valida HIPÓTESIS: aquí rechaza la
# nuestra, y un resultado negativo también es un hallazgo. Sube k (3, 4...)
# o agrupa SIN z-normalizar (nivel invierno/verano) y compara.'''))

    cells.append(("md", r"""## 6. Registro en MLflow"""))

    cells.append(("code", r'''setup_mlflow("module3-06-clustering", backend="dagshub")

figs = {}
if km_labels is not None:
    figs["plots/clusters.png"] = fig_clusters
log_and_register(
    run_name="dtw-load-profiles",
    params={"n_days": int(N), "k": 2, "metric": "dtw",
            "znorm": True, "dataset": "uci-household-power"},
    metrics={**({"ARI_kmeans_weekend": float(ari_km)}
               if km_labels is not None else {}),
             "ARI_hierarchical_weekend": float(ari_h)},
    tags={"notebook": "06_timeseries_clustering", "familia": "unsupervised"},
    figures=figs,
)'''))

    cells.append(("md", r"""## Resumen

- Cada día del hogar es un **perfil de carga** de 24 puntos; agruparlos revela
  arquetipos de comportamiento.
- **z-normaliza** para agrupar por *forma* y no por nivel; usa **DTW** (no
  Euclidiana) para tolerar rutinas desfasadas — recurrencia
  $D(i,j) = c(i,j) + \min\{\dots\}$ con camino de deformación.
- Herramientas: `tslearn` **TimeSeriesKMeans (DTW + DBA)** o el **jerárquico
  de scipy** sobre la matriz DTW (el dendrograma ayuda a elegir $k$).
- Valida contra etiquetas conocidas con el **Índice de Rand Ajustado** — y
  acepta el veredicto: aquí el ARI ≈ 0 **rechaza** la hipótesis laboral/finde
  (DTW + z-norm hacen muy parecidas ambas rutinas) y lo que emerge es el
  arquetipo *día atípico / vacaciones*. Un resultado negativo bien validado
  también es un hallazgo.

**Fin del Módulo 3.** El recorrido completo: descomposición y baselines →
suavizamiento exponencial y SARIMA → ingeniería de variables → XGBoost →
ensambles → clustering, todo sobre el mismo dataset real y versionado en
MLflow."""))

    build("06_timeseries_clustering.ipynb", cells)


if __name__ == "__main__":
    nb01_decomposition()
    nb02_arima()
    nb03_features()
    nb04_xgboost()
    nb05_ensembles()
    nb06_clustering()
    print("\nTodos los notebooks generados.")
