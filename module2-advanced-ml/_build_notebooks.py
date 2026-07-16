#!/usr/bin/env python3
"""
_build_notebooks.py
===================

Genera los seis notebooks del Módulo 2 usando ``nbformat`` (nunca escritos a
mano como JSON). Ejecutar con::

    python3 _build_notebooks.py

Cada notebook se ensambla a partir de una lista de tuplas (kind, source), donde
kind es ``"md"`` (markdown) o ``"code"``.
"""

import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "notebooks")
os.makedirs(OUT, exist_ok=True)


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
        "language_info": {"name": "python", "version": "3.10"},
    }
    path = os.path.join(OUT, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {path}")


# Celda de arranque común inyectada cerca del inicio de cada notebook para que el
# paquete `utils` sea importable y MLflow quede configurado.
BOOTSTRAP = r"""
import os, sys, warnings
warnings.filterwarnings("ignore")

# Hacemos importable utils/ tanto si el notebook corre desde notebooks/ como
# desde la raíz del repositorio.
_here = os.getcwd()
for cand in (os.path.join(_here, "..", "utils"), os.path.join(_here, "utils"),
             os.path.join(_here, "..", "..", "module2-advanced-ml", "utils")):
    cand = os.path.abspath(cand)
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow_helpers import setup_mlflow, log_and_register, register_best_run, registry_available

np.random.seed(42)
print("Versión de MLflow:", mlflow.__version__)
"""

# ===========================================================================
# 01 — REGULARIZACIÓN
# ===========================================================================
nb01 = [
    ("md", r"""# 01 — Regularización (Ridge, Lasso, ElasticNet)

**Módulo 2 · Algoritmos avanzados de ML (+ MLflow)**

En este notebook estudiamos la **regularización**: una familia de técnicas que
limitan la complejidad de un modelo para mejorar su **generalización**. Vemos la
intuición y las fórmulas clave de **Ridge (L2)**, **Lasso (L1)** y
**ElasticNet**, entrenamos las tres sobre el dataset **California housing**,
comparamos sus coeficientes y su error en validación cruzada, y registramos cada
experimento con **MLflow**, guardando el mejor modelo en el **Model Registry**.
"""),
    ("md", r"""## 1. ¿Por qué regularizar? Sobreajuste y el compromiso sesgo–varianza

**Idea intuitiva:** un modelo demasiado flexible memoriza el ruido de los datos
de entrenamiento y luego falla con datos nuevos (sobreajuste); uno demasiado
simple no capta ni la señal (subajuste). La regularización busca el punto medio.

Formalmente, para el error cuadrático el error de generalización se descompone:

$$
\mathbb{E}\big[(y - \hat f(x))^2\big]
= \underbrace{\text{sesgo}^2}_{\text{modelo muy simple}}
+ \underbrace{\text{varianza}}_{\text{modelo muy flexible}}
+ \underbrace{\sigma^2}_{\text{ruido irreducible}}.
$$

La **regularización** añade una *penalización* sobre el tamaño de los parámetros:

$$
\min_{\beta}\; \underbrace{L(\beta)}_{\text{ajuste a los datos}} \; + \; \lambda \, R(\beta),
$$

cambiando un poco más de sesgo por una gran reducción de varianza. El
hiperparámetro $\lambda$ (llamado `alpha` en scikit-learn) controla la fuerza.
"""),
    ("code", BOOTSTRAP),
    ("code", r"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
feature_names = list(X.columns)
print("Dimensiones:", X.shape, "| objetivo = valor mediano de la vivienda ($100k)")
X.head()
"""),
    ("code", r"""
# La regularización penaliza la MAGNITUD de los coeficientes, así que las
# variables deben estar en la misma escala. Estandarizamos (media 0, desv. 1).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
print("Train:", X_train_s.shape, "Test:", X_test_s.shape)
"""),
    ("md", r"""## 2. Tracking de experimentos y Model Registry: por qué importan

**El tracking de experimentos** registra *cada* ejecución: sus parámetros (p. ej.
`alpha`), sus métricas (p. ej. RMSE en test), la versión del código y los
artefactos (el modelo entrenado, gráficas). Sin él, comparar decenas de modelos
es adivinar y los resultados no son reproducibles.

**El Model Registry** es un catálogo versionado de *modelos* (no de ejecuciones).
Permite promover una versión concreta por etapas (Staging → Production), adjuntar
metadatos y servir una única fuente de verdad a los sistemas aguas abajo.

**MLflow** nos da ambas cosas. Lo configuramos una vez con nuestro helper, que
apunta al tracking server (`http://localhost:5000`) o cae a un store local.
"""),
    ("code", r"""
setup_mlflow("module2-01-regularization", backend="dagshub")
"""),
    ("md", r"""## 3. Regresión Ridge (L2)

**Intuición:** Ridge encoge todos los coeficientes hacia cero (sin anularlos),
lo que estabiliza el modelo cuando hay variables muy correlacionadas.

Resuelve:

$$
\hat\beta_{\text{ridge}} = \arg\min_\beta \; \|y - X\beta\|_2^2 + \alpha\,\|\beta\|_2^2,
\qquad \|\beta\|_2^2 = \sum_j \beta_j^2 .
$$

Tiene **solución cerrada** (siempre invertible para $\alpha>0$):

$$
\hat\beta_{\text{ridge}} = (X^\top X + \alpha I)^{-1} X^\top y .
$$

El término $\alpha I$ *encoge* los coeficientes y estabiliza la inversa cuando
las variables son colineales. Visualicemos el **camino de encogimiento** según
crece $\alpha$.
"""),
    ("code", r"""
from sklearn.linear_model import Ridge

alphas = np.logspace(-2, 4, 50)
coefs = []
for a in alphas:
    m = Ridge(alpha=a).fit(X_train_s, y_train)
    coefs.append(m.coef_)
coefs = np.array(coefs)

plt.figure(figsize=(8, 5))
for j, name in enumerate(feature_names):
    plt.plot(alphas, coefs[:, j], label=name)
plt.xscale("log")
plt.xlabel(r"$\alpha$ (fuerza de regularización)")
plt.ylabel("valor del coeficiente")
plt.title("Camino de encogimiento de Ridge")
plt.legend(fontsize=8); plt.axhline(0, color="k", lw=0.5)
plt.tight_layout(); plt.show()
"""),
    ("md", r"""## 4. Regresión Lasso (L1): esparsidad y selección de variables

**Intuición:** Lasso no sólo encoge, sino que lleva algunos coeficientes a
**exactamente cero**, eligiendo automáticamente qué variables conservar.

Reemplaza la penalización L2 por la **norma L1**:

$$
\hat\beta_{\text{lasso}} = \arg\min_\beta \; \|y - X\beta\|_2^2 + \alpha\,\|\beta\|_1,
\qquad \|\beta\|_1 = \sum_j |\beta_j| .
$$

No tiene forma cerrada (la L1 no es diferenciable en 0); se resuelve por
descenso por coordenadas.

### Geometría: por qué L1 → esparsidad y L2 → encogimiento
Las regiones de restricción son $\|\beta\|_1 \le t$ (un rombo) frente a
$\|\beta\|_2 \le t$ (un círculo). Los contornos elípticos del error cuadrático
tocan primero las **esquinas** del rombo —que caen sobre los ejes (algún
$\beta_j = 0$)— mientras que tocan el círculo suave en un punto cualquiera (sin
ceros exactos).
"""),
    ("code", r"""
from sklearn.linear_model import Lasso

alphas_l = np.logspace(-3, 1, 50)
coefs_l = []
for a in alphas_l:
    m = Lasso(alpha=a, max_iter=10000).fit(X_train_s, y_train)
    coefs_l.append(m.coef_)
coefs_l = np.array(coefs_l)

plt.figure(figsize=(8, 5))
for j, name in enumerate(feature_names):
    plt.plot(alphas_l, coefs_l[:, j], label=name)
plt.xscale("log")
plt.xlabel(r"$\alpha$"); plt.ylabel("valor del coeficiente")
plt.title("Camino de Lasso: los coeficientes llegan a 0 exacto (esparsidad)")
plt.legend(fontsize=8); plt.axhline(0, color="k", lw=0.5)
plt.tight_layout(); plt.show()
"""),
    ("code", r"""
# Ilustramos la geometría de la restricción L1 (rombo) vs L2 (círculo)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
theta = np.linspace(0, 2*np.pi, 400)
for ax, title, region in zip(
    axes, ["Bola L2 (Ridge)", "Bola L1 (Lasso)"], ["l2", "l1"]):
    if region == "l2":
        ax.plot(np.cos(theta), np.sin(theta), 'b')
    else:
        ax.plot([1,0,-1,0,1],[0,1,0,-1,0],'g')
    # contornos de la pérdida (elipses centradas fuera del origen)
    cx, cy = 1.6, 1.1
    for r in [0.4, 0.8, 1.2, 1.6]:
        ax.plot(cx + r*1.4*np.cos(theta), cy + r*0.8*np.sin(theta), 'r', lw=0.6)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-2.5, 3); ax.set_ylim(-2, 2.5); ax.set_aspect('equal')
    ax.set_title(title)
plt.suptitle("Dónde tocan los contornos de la pérdida (rojo) a las restricciones")
plt.tight_layout(); plt.show()
"""),
    ("md", r"""## 5. ElasticNet: la penalización combinada

**Intuición:** ElasticNet mezcla L1 y L2, así que conserva la esparsidad de Lasso
pero reparte mejor el peso entre **grupos de variables correlacionadas** (Lasso
tiende a quedarse con una arbitrariamente).

$$
\hat\beta = \arg\min_\beta \; \frac{1}{2n}\|y - X\beta\|_2^2
+ \alpha\Big( \rho\,\|\beta\|_1 + \tfrac{1-\rho}{2}\,\|\beta\|_2^2 \Big),
$$

donde `l1_ratio` $= \rho \in [0,1]$ interpola entre Ridge ($\rho=0$) y Lasso
($\rho=1$).
"""),
    ("md", r"""## 6. Entrenar las tres, comparar y registrar en MLflow

Comparamos Ridge, Lasso y ElasticNet por RMSE en validación cruzada de 5 folds
sobre el set de entrenamiento, y reportamos RMSE / $R^2$ en test. Cada uno se
registra como una ejecución de MLflow; el mejor (menor RMSE en CV) se registra.
"""),
    ("code", r"""
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_true, y_pred):
    # Funciona en distintas versiones de sklearn (squared= se quitó en 1.8)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate(model, name, params):
    cv_rmse = -cross_val_score(model, X_train_s, y_train, cv=5,
                               scoring="neg_root_mean_squared_error").mean()
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    test_rmse = rmse(y_test, pred)
    test_r2 = r2_score(y_test, pred)
    n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-8))
    metrics = {"cv_rmse": cv_rmse, "test_rmse": test_rmse,
               "test_r2": test_r2, "n_nonzero_coef": n_nonzero}
    run_id = log_and_register(run_name=name, params=params, metrics=metrics,
                              model=model, flavor="sklearn",
                              tags={"family": "linear", "algo": name})
    print(f"{name:12s} | CV RMSE {cv_rmse:.4f} | test RMSE {test_rmse:.4f} | "
          f"R2 {test_r2:.4f} | no nulos {n_nonzero}")
    return metrics, model

results = {}
results["Ridge"], ridge = evaluate(Ridge(alpha=1.0), "Ridge", {"alpha": 1.0})
results["Lasso"], lasso = evaluate(Lasso(alpha=0.01, max_iter=10000), "Lasso", {"alpha": 0.01})
results["ElasticNet"], enet = evaluate(
    ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
    "ElasticNet", {"alpha": 0.01, "l1_ratio": 0.5})
"""),
    ("code", r"""
# Comparación de coeficientes
comp = pd.DataFrame({"feature": feature_names,
                     "Ridge": ridge.coef_, "Lasso": lasso.coef_,
                     "ElasticNet": enet.coef_})
comp
"""),
    ("code", r"""
comp.set_index("feature").plot.bar(figsize=(9,5))
plt.title("Comparación de coeficientes entre regularizadores")
plt.ylabel("coeficiente"); plt.axhline(0, color="k", lw=0.5)
plt.tight_layout(); plt.show()
"""),
    ("code", r"""
# Registramos el mejor modelo (menor RMSE en CV) en el Model Registry
register_best_run("module2-01-regularization", metric="cv_rmse",
                  registered_model_name="california-housing-regularized",
                  mode="min")
"""),
    ("md", r"""## 7. Resumen

| Método | Penalización | Efecto | ¿Forma cerrada? |
|---|---|---|---|
| Ridge | $\alpha\|\beta\|_2^2$ | encoge todos, ninguno cero | sí |
| Lasso | $\alpha\|\beta\|_1$ | esparso, selección de variables | no (desc. por coordenadas) |
| ElasticNet | mezcla vía `l1_ratio` | esparso + maneja correlación | no |

La regularización es tu primera línea de defensa contra el sobreajuste. Ajusta
`alpha` (y `l1_ratio`) por validación cruzada, y deja que MLflow guarde la
evidencia. Abre la UI en **http://localhost:5000** para inspeccionar runs y el
registry.
"""),
]

# ===========================================================================
# 02 — ENSEMBLES
# ===========================================================================
nb02 = [
    ("md", r"""# 02 — Métodos de ensamble (Bagging, Boosting, RF, XGBoost, LightGBM, CatBoost)

**Módulo 2 · Algoritmos avanzados de ML (+ MLflow)**

Los **ensembles** combinan muchos aprendices *débiles* en uno *fuerte*. Vemos la
intuición de **bagging vs boosting**, la ponemos en práctica con
**`BaggingClassifier`** y un **Random Forest**, comparamos las tres librerías
modernas de gradient boosting —**XGBoost**, **LightGBM**, **CatBoost**—
explicando *en qué se diferencian*, y cerramos con los meta-ensembles de
scikit-learn: **`VotingClassifier`** y **`StackingClassifier`**. Todo se
registra en **MLflow** y el mejor modelo se guarda en el registry.
"""),
    ("md", r"""## 1. La visión sesgo–varianza de los ensembles

**Intuición:** promediar muchos modelos reduce la varianza, sobre todo si esos
modelos cometen errores *poco correlacionados*.

Si promediamos $B$ modelos, cada uno con varianza $\sigma^2$ y correlación por
pares $\rho$, la varianza del promedio es:

$$
\operatorname{Var}\Big(\tfrac{1}{B}\sum_b f_b\Big)
= \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .
$$

Dos palancas para reducir varianza: **más modelos** ($B\uparrow$) y **menos
correlación** ($\rho\downarrow$). Esa es la idea del **bagging**.

### Bagging (Bootstrap AGGregatING)
- Se sacan $B$ muestras **bootstrap** (muestrear $n$ filas *con reemplazo*).
- Se entrena un modelo por muestra; se **promedia** (regresión) o se **vota**
  (clasificación).
- Reduce la **varianza**, deja el sesgo casi igual. Los modelos son
  independientes → trivialmente paralelizable. **Random Forest** = árboles
  embolsados + decorrelación extra muestreando variables en cada split.

### Boosting
- Entrena modelos **secuencialmente**, cada nuevo enfocado en los *errores
  residuales* del ensemble actual.
- El gradient boosting ajusta cada nuevo árbol al **gradiente negativo** de la
  pérdida:
$$
F_{m}(x) = F_{m-1}(x) + \nu\, h_m(x),\qquad
h_m \approx -\frac{\partial L(y, F_{m-1}(x))}{\partial F_{m-1}(x)},
$$
donde $\nu$ es la tasa de aprendizaje. Reduce el **sesgo** (y la varianza); más
potente pero más propenso al sobreajuste y secuencial por naturaleza.
"""),
    ("code", BOOTSTRAP),
    ("code", r"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape,
      "| clases:", dict(pd.Series(y).value_counts()))
setup_mlflow("module2-02-ensembles", backend="dagshub")
"""),
    ("code", r"""
# Un pequeño helper para puntuar + registrar un clasificador ya entrenado
def score_and_log(model, name, params, flavor="sklearn"):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"accuracy": accuracy_score(y_test, pred),
               "f1": f1_score(y_test, pred),
               "roc_auc": roc_auc_score(y_test, proba)}
    log_and_register(run_name=name, params=params, metrics=metrics,
                     model=model, flavor=flavor, tags={"algo": name})
    print(f"{name:14s} | acc {metrics['accuracy']:.4f} | "
          f"f1 {metrics['f1']:.4f} | auc {metrics['roc_auc']:.4f}")
    return metrics
"""),
    ("md", r"""## 2. Bagging en la práctica: `BaggingClassifier`

Antes de saltar al Random Forest, veamos el bagging *puro* con el
`BaggingClassifier` de scikit-learn: $B$ copias del **mismo estimador base**
(aquí un árbol de decisión profundo), cada una entrenada sobre una muestra
bootstrap, que luego **votan**.

Dos cosas que observar:

1. **Reducción de varianza:** un árbol profundo solo memoriza el training set
   (varianza alta); el mismo árbol embolsado $B$ veces generaliza mucho mejor,
   sin cambiar el estimador base.
2. **Out-of-bag (OOB):** cada muestra bootstrap deja fuera ≈36.8% de las filas
   ($e^{-1}$); evaluar cada árbol sobre *sus* filas excluidas da una estimación
   de generalización "gratis", sin tocar el test set (`oob_score=True`).
"""),
    ("code", r"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Referencia: UN solo árbol profundo — mucha varianza
tree_params = {"random_state": 42}
m_tree = score_and_log(DecisionTreeClassifier(**tree_params), "DecisionTree", tree_params)

# El mismo árbol, embolsado 300 veces sobre muestras bootstrap
bag_params = {"n_estimators": 300, "oob_score": True,
              "random_state": 42, "n_jobs": -1}
bag = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), **bag_params)
m_bag = score_and_log(bag, "Bagging-tree", bag_params)
print(f"OOB score (estimación de generalización sin usar el test): {bag.oob_score_:.4f}")
"""),
    ("md", r"""## 3. Random Forest

Un Random Forest hace crecer muchos árboles profundos, cada uno sobre una muestra
bootstrap y eligiendo cada split de un subconjunto aleatorio de variables
(`max_features`). Esa aleatoriedad extra *decorrelaciona* los árboles
($\rho\downarrow$), de modo que promediar sus predicciones reduce mucho la
varianza — es `BaggingClassifier` + muestreo de variables por split.
"""),
    ("code", r"""
from sklearn.ensemble import RandomForestClassifier
rf_params = {"n_estimators": 300, "max_depth": None,
             "max_features": "sqrt", "random_state": 42, "n_jobs": -1}
m_rf = score_and_log(RandomForestClassifier(**rf_params), "RandomForest", rf_params)
"""),
    ("code", r"""
# Importancia de variables según el bosque
rf = RandomForestClassifier(**rf_params).fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()[-12:]
imp.plot.barh(figsize=(7,5)); plt.title("Random Forest — importancia de variables (top)")
plt.tight_layout(); plt.show()
"""),
    ("md", r"""## 4. Las tres librerías de gradient boosting

Las tres minimizan un objetivo regularizado de la forma:

$$
\mathcal{L} = \sum_i \ell(y_i, \hat y_i) + \sum_k \Omega(f_k),\qquad
\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \|w\|^2,
$$

(pérdida + penalización de complejidad sobre cada árbol $f_k$ con $T$ hojas y
pesos de hoja $w$). Se diferencian en **cómo crecen los árboles** y en **cómo
manejan los datos y las categorías**.
"""),
    ("code", r"""
import xgboost as xgb
from xgboost import XGBClassifier
xgb_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
              "subsample": 0.9, "colsample_bytree": 0.9,
              "reg_lambda": 1.0, "eval_metric": "logloss",
              "random_state": 42, "n_jobs": -1}
m_xgb = score_and_log(XGBClassifier(**xgb_params), "XGBoost", xgb_params, flavor="xgboost")
"""),
    ("code", r"""
from lightgbm import LGBMClassifier
lgb_params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
              "subsample": 0.9, "colsample_bytree": 0.9,
              "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1, "verbose": -1}
m_lgb = score_and_log(LGBMClassifier(**lgb_params), "LightGBM", lgb_params, flavor="lightgbm")
"""),
    ("code", r"""
from catboost import CatBoostClassifier
cat_params = {"iterations": 300, "learning_rate": 0.05, "depth": 4,
              "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": False}
m_cat = score_and_log(CatBoostClassifier(**cat_params), "CatBoost", cat_params, flavor="catboost")
"""),
    ("md", r"""## 5. En qué se diferencian XGBoost, LightGBM y CatBoost

| Aspecto | **XGBoost** | **LightGBM** | **CatBoost** |
|---|---|---|---|
| Crecimiento del árbol | **Por niveles (depth-wise)** — crece todos los nodos de un nivel | **Por hoja (best-first)** — divide la hoja de mayor ganancia | Árboles **simétricos / oblivious** — misma variable+umbral por nivel |
| Búsqueda de splits | Pre-ordenado e **histograma aproximado** | **Histograma** de las variables | Histograma (oblivious) |
| Aceleración en big-data | Consciente de esparsidad, weighted quantile sketch | **GOSS** + **EFB** (bundling de variables exclusivas) | Ordered boosting sobre permutaciones |
| Variables categóricas | Necesitan encoding (one-hot / ordinal) | Soporte nativo (codificadas como enteros) | **Nativo**, con **target statistics ordenadas** (sin fuga) |
| Control de sobreajuste | $\gamma$, $\lambda$, `max_depth`, subsample | num_leaves, min_data_in_leaf, bagging | ordered boosting reduce el sesgo de **prediction shift** |
| Mejor cuando | default robusto, tabular, tamaño medio | **datasets grandes**, muchas variables, velocidad | **muchas categóricas**, buenos defaults |

**En palabras:**
- **XGBoost** popularizó el gradient boosting *regularizado*. Crece los árboles
  **por niveles** y añade la penalización $\Omega$, más shrinkage y submuestreo de
  columnas. Un default sólido y robusto.
- **LightGBM** crece los árboles **por hoja** (siempre la de mayor ganancia), lo
  que da árboles más profundos y precisos por iteración pero puede sobreajustar en
  datos pequeños — contrólalo con `num_leaves`/`min_data_in_leaf`. Usa
  **histogramas** más **GOSS** (mantener ejemplos de gradiente alto, submuestrear
  los de gradiente bajo) y **EFB** (agrupar variables esparsas mutuamente
  excluyentes), siendo muy **rápido en datos grandes y de alta dimensión**.
- **CatBoost** ataca dos problemas: (1) la **fuga de información** al codificar
  categóricas —resuelta con **target statistics ordenadas** sobre permutaciones
  aleatorias—; y (2) el **prediction shift** del boosting estándar —resuelto con
  **ordered boosting**. Usa **árboles simétricos (oblivious)**, rápidos de evaluar
  y que actúan como regularización. Mejor con **muchas variables categóricas**.
"""),
    ("md", r"""## 6. Tabla comparativa y registro del mejor"""),
    ("code", r"""
bench = pd.DataFrame({"DecisionTree": m_tree, "Bagging-tree": m_bag,
                      "RandomForest": m_rf, "XGBoost": m_xgb,
                      "LightGBM": m_lgb, "CatBoost": m_cat}).T
bench = bench[["accuracy", "f1", "roc_auc"]].sort_values("roc_auc", ascending=False)
display(bench.style.background_gradient(cmap="Greens"))
"""),
    ("code", r"""
bench["roc_auc"].plot.barh(figsize=(7,4)); plt.xlim(0.95, 1.0)
plt.title("ROC-AUC por ensemble en el test de breast cancer")
plt.tight_layout(); plt.show()
"""),
    ("code", r"""
register_best_run("module2-02-ensembles", metric="roc_auc",
                  registered_model_name="breast-cancer-ensemble", mode="max")
"""),
    ("md", r"""## 7. Clasificadores por votación: el meta-ensemble más simple

Antes del combinador aprendido del stacking, la forma más simple de combinar
varios estimadores base **ya entrenados y diversos** es **votar**. No hay
meta-modelo: la regla de combinación es **fija**.

### Voto duro (mayoría)
Cada clasificador base emite un voto y el ensemble predice la clase más común:
$$
\hat y = \operatorname{moda}\{c_1(x),\, c_2(x),\, \dots,\, c_M(x)\}.
$$

### Voto suave (promedio de probabilidades)
Se promedian las probabilidades por clase (opcionalmente con pesos $w_m$) y se
toma el arg-máx:
$$
\hat y = \arg\max_k \sum_{m=1}^{M} w_m\, p_{m,k}(x).
$$
El voto suave suele ganar al duro porque usa la *confianza*, no sólo la etiqueta
ganadora — pero requiere `predict_proba` razonablemente **calibrado** (p. ej.
envolver `SVC` con `probability=True`). Los pesos permiten confiar más en los
modelos fuertes.

### Por qué importa la diversidad
La votación sólo ayuda si los aprendices base cometen **errores decorrelacionados**:
si todos fallan en las mismas filas, promediar no cambia nada. Combinar un RF
basado en árboles, un SVC de margen y un modelo lineal logístico hace que los
errores se cancelen en parte.

### En qué difiere la votación del stacking y del bagging
- **vs stacking:** la votación usa una regla **fija** (moda / promedio
  ponderado); el stacking *aprende* un meta-modelo sobre las predicciones base.
- **vs bagging:** el bagging re-muestrea el **mismo** algoritmo sobre muestras
  bootstrap; la votación combina algoritmos **distintos** sobre todos los datos.
"""),
    ("code", r"""
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Aprendices base diversos. Los sensibles a la escala (LogReg, SVC) van en
# pipelines con StandardScaler, igual que en la sección de Stacking de abajo.
vote_estimators = [
    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("rf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                  random_state=42, n_jobs=-1)),
    ("svc", make_pipeline(StandardScaler(),
                          SVC(C=1.0, kernel="rbf", probability=True, random_state=42))),
]

# Voto duro: mayoría sobre las etiquetas predichas.
vote_hard = VotingClassifier(estimators=vote_estimators, voting="hard", n_jobs=-1)
"""),
    ("code", r"""
# VotingClassifier con voting="hard" no tiene predict_proba, así que score_and_log
# (que necesita probabilidades) no se puede reusar tal cual. Lo puntuamos directo
# y registramos la ejecución con los mismos helpers.
vote_hard.fit(X_train, y_train)
hard_pred = vote_hard.predict(X_test)
m_vote_hard = {"accuracy": accuracy_score(y_test, hard_pred),
               "f1": f1_score(y_test, hard_pred),
               "roc_auc": np.nan}  # indefinido: el voto duro da etiquetas, no probs
log_and_register(run_name="Voting-hard",
                 params={"voting": "hard", "base_learners": "lr+rf+svc"},
                 metrics={"accuracy": m_vote_hard["accuracy"], "f1": m_vote_hard["f1"]},
                 model=vote_hard, flavor="sklearn", tags={"algo": "Voting-hard"})
print(f"{'Voting-hard':14s} | acc {m_vote_hard['accuracy']:.4f} | "
      f"f1 {m_vote_hard['f1']:.4f} | auc   n/a")
"""),
    ("code", r"""
# Voto suave: promedia las probabilidades predichas. Reusa score_and_log porque
# expone predict_proba.
vote_soft = VotingClassifier(estimators=vote_estimators, voting="soft", n_jobs=-1)
m_vote_soft = score_and_log(vote_soft, "Voting-soft",
                            {"voting": "soft", "base_learners": "lr+rf+svc"})

# Voto suave ponderado: apoyarse más en los aprendices fuertes (RF + SVC).
vote_soft_w = VotingClassifier(estimators=vote_estimators, voting="soft",
                               weights=[1, 2, 2], n_jobs=-1)
m_vote_soft_w = score_and_log(vote_soft_w, "Voting-soft-weighted",
                              {"voting": "soft", "weights": "1,2,2",
                               "base_learners": "lr+rf+svc"})
"""),
    ("code", r"""
# Comparamos las variantes de votación con sus propios aprendices base y con los
# modelos de boosting.
from sklearn.base import clone

vote_base_scores = {}
for bname, bmodel in vote_estimators:
    clf = clone(bmodel).fit(X_train, y_train)
    bproba = clf.predict_proba(X_test)[:, 1]
    bpred = (bproba >= 0.5).astype(int)
    vote_base_scores[bname] = {"accuracy": accuracy_score(y_test, bpred),
                               "f1": f1_score(y_test, bpred),
                               "roc_auc": roc_auc_score(y_test, bproba)}

vote_cmp = pd.DataFrame({**vote_base_scores,
                         "Voting-hard": m_vote_hard,
                         "Voting-soft": m_vote_soft,
                         "Voting-soft-weighted": m_vote_soft_w,
                         "XGBoost": m_xgb, "LightGBM": m_lgb,
                         "CatBoost": m_cat}).T
vote_cmp = vote_cmp[["accuracy", "f1", "roc_auc"]].sort_values("f1", ascending=False)

best_single_f1 = max(v["f1"] for v in vote_base_scores.values())
print(f"Mejor F1 de un base individual: {best_single_f1:.4f} | "
      f"F1 Voting-soft: {m_vote_soft['f1']:.4f}")
display(vote_cmp.style.background_gradient(cmap="Purples"))
"""),
    ("code", r"""
# Los runs de voto suave se registraron con roc_auc, así que re-ejecutar la
# promoción del registry elige un modelo de votación si ahora supera a los demás.
register_best_run("module2-02-ensembles", metric="roc_auc",
                  registered_model_name="breast-cancer-ensemble", mode="max")
"""),
    ("md", r"""## 8. Stacking y blending: el pilar del combinador aprendido

El bagging promedia aprendices **independientes** (varianza$\downarrow$) y el
boosting encadena aprendices **secuenciales** (sesgo$\downarrow$). El
**stacked generalization** (stacking) va más allá de la votación de regla fija:
entrena varios **aprendices base diversos** y luego un **meta-aprendiz** que
aprende *cómo combinar sus predicciones*.

### Por qué funciona
Un promedio simple trata a todos los modelos igual. Un meta-aprendiz, en cambio,
aprende **pesos guiados por los datos** (e incluso combinaciones no lineales)
sobre las salidas de los modelos base. La ganancia viene de la **decorrelación de
errores**: si los modelos base cometen errores *distintos*, se cancelan en parte,
y el meta-aprendiz se apoya en el modelo fiable en cada región. Si todos los
modelos base fueran idénticos, el stacking no aportaría nada — **la diversidad es
el combustible**.

### La trampa de la fuga → predicciones out-of-fold (OOF)
El meta-aprendiz debe entrenarse con predicciones de los modelos base sobre
ejemplos que estos **no vieron durante su entrenamiento**. Si los entrenáramos
sobre todo el set y luego les diéramos sus predicciones *in-sample*, estas serían
demasiado optimistas (ya "vieron la respuesta"), el meta-aprendiz sobreajustaría
y el rendimiento en test colapsaría. La solución son las **predicciones
out-of-fold (OOF)** vía $k$-fold:

1. Dividir el set de entrenamiento en $k$ folds.
2. Para cada fold, entrenar los modelos base en los otros $k-1$ folds y predecir
   el fold retenido. Concatenando se obtiene una **predicción OOF limpia por
   fila** — ninguna fila la predijo un modelo que se entrenó con ella.
3. Entrenar el meta-aprendiz sobre la matriz OOF $Z \in \mathbb{R}^{n \times M}$
   (filas = ejemplos, columnas = $M$ modelos base).
4. Re-entrenar cada modelo base con **todo** el set para inferir sobre datos
   nuevos.

$$
\hat y = g\big(f_1(x),\, f_2(x),\, \dots,\, f_M(x)\big),
$$

donde $f_m$ son los aprendices base y $g$ el meta-aprendiz entrenado sobre $Z$.

### Blending vs stacking
- **Stacking** usa predicciones **OOF** de $k$-fold — cada fila contribuye a las
  meta-features, así que es eficiente en datos (pero entrena los base $k$ veces).
- **Blending** usa un único **holdout**: los base entrenan en la parte A, el
  meta-aprendiz entrena con sus predicciones para una parte B retenida. Más simple
  y rápido, pero el meta-aprendiz ve menos filas.

El `StackingClassifier` de scikit-learn implementa la variante de stacking: pasar
`cv=5` genera las meta-features con OOF de 5 folds internamente.
"""),
    ("code", r"""
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Aprendices base diversos: un ensemble de árboles embolsado, un clasificador de
# margen y uno basado en distancia. SVC/kNN necesitan escalado → pipelines.
base_estimators = [
    ("rf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                  random_state=42, n_jobs=-1)),
    ("svc", make_pipeline(StandardScaler(),
                          SVC(C=1.0, kernel="rbf", probability=True, random_state=42))),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))),
]

# Meta-aprendiz de regresión logística entrenado con predicciones OOF de 5 folds.
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    cv=5,
    n_jobs=-1,
)
stack_params = {"final_estimator": "LogisticRegression", "cv": 5,
                "base_learners": "rf+svc+knn", "stack_method": "predict_proba"}
m_stack = score_and_log(stack, "Stacking", stack_params)
"""),
    ("code", r"""
# ¿Cómo se compara el stacking con sus propios aprendices base y los de boosting?
# (Puntuamos cada base por separado para una comparación justa.)
from sklearn.base import clone

base_scores = {}
for bname, bmodel in base_estimators:
    clf = clone(bmodel).fit(X_train, y_train)
    bproba = clf.predict_proba(X_test)[:, 1]
    bpred = (bproba >= 0.5).astype(int)
    base_scores[bname] = {"accuracy": accuracy_score(y_test, bpred),
                          "f1": f1_score(y_test, bpred),
                          "roc_auc": roc_auc_score(y_test, bproba)}

stack_cmp = pd.DataFrame({**base_scores,
                          "Stacking": m_stack,
                          "XGBoost": m_xgb, "LightGBM": m_lgb,
                          "CatBoost": m_cat}).T
stack_cmp = stack_cmp[["accuracy", "f1", "roc_auc"]].sort_values("f1", ascending=False)

best_single_f1 = max(v["f1"] for v in base_scores.values())
print(f"Mejor F1 de un base individual: {best_single_f1:.4f} | "
      f"F1 Stacking: {m_stack['f1']:.4f}")
display(stack_cmp.style.background_gradient(cmap="Blues"))
"""),
    ("code", r"""
# El stacking se registró como un run del mismo experimento, así que re-ejecutar
# la promoción del registry lo elige si ahora tiene el mejor ROC-AUC.
register_best_run("module2-02-ensembles", metric="roc_auc",
                  registered_model_name="breast-cancer-ensemble", mode="max")
"""),
    ("md", r"""## 9. Resumen

| Familia | Método(s) | Idea central | Reduce |
|---|---|---|---|
| **Bagging** | BaggingClassifier / Random Forest | promediar árboles independientes y decorrelacionados | varianza |
| **Boosting** | XGBoost / LightGBM / CatBoost | corregir residuos secuencialmente | sesgo (+varianza) |
| **Meta-ensembling** | **Voting** y **Stacking** | combinar aprendices base diversos | error residual |

- **Bagging** (Random Forest) reduce la **varianza** con árboles independientes y
  decorrelacionados — vergonzosamente paralelo.
- **Boosting** reduce el **sesgo** corrigiendo residuos en secuencia.
- Las tres librerías de boosting comparten el objetivo GBM regularizado pero
  difieren en **crecimiento del árbol** (niveles / hoja / simétrico), **trucos de
  velocidad** (GOSS+EFB) y **manejo de categóricas** (target statistics ordenadas).
- El **meta-ensembling** combina aprendices **diversos**. La **votación** usa una
  regla **fija** — mayoría (duro) o promedio ponderado de probabilidades (suave);
  el **stacking** entrena un **meta-aprendiz** sobre predicciones **out-of-fold**
  para evitar la fuga, con el **blending** como variante holdout más barata.
- Elige **LightGBM** para datos grandes, **CatBoost** para muchas categóricas,
  **XGBoost** como default robusto, empieza el meta-ensembling con **votación** y
  recurre al **stacking** para exprimir el último punto — siempre con MLflow.
"""),
]

# ===========================================================================
# 03 — SVM
# ===========================================================================
nb03 = [
    ("md", r"""# 03 — Máquinas de soporte vectorial (SVM)

**Módulo 2 · Algoritmos avanzados de ML (+ MLflow)**

Las SVM buscan el hiperplano separador de **margen máximo**. Vemos la intuición y
las fórmulas del problema **primal** y **dual**, la **pérdida hinge** y el
parámetro de margen blando $C$, explicamos el **truco del kernel** (lineal, RBF,
polinómico), visualizamos las fronteras de decisión sobre `make_moons` y ajustamos
$C$/$\gamma$ — registrando todo en **MLflow**.
"""),
    ("md", r"""## 1. El clasificador de margen máximo

**Intuición:** entre todas las rectas que separan dos clases, la SVM elige la que
deja el **mayor margen** (la mayor distancia a los puntos más cercanos). Un margen
ancho generaliza mejor.

Para datos separables con etiquetas $y_i \in \{-1, +1\}$, un hiperplano es
$w^\top x + b = 0$. El margen geométrico es $1/\|w\|$, así que resolvemos el
**primal**:

$$
\min_{w,b}\;\; \tfrac{1}{2}\|w\|^2
\quad\text{s.a.}\quad y_i(w^\top x_i + b) \ge 1 \;\; \forall i .
$$

### Margen blando (datos no separables)
Los datos reales se solapan. Introducimos holgura $\xi_i \ge 0$ y una penalización
$C$:

$$
\min_{w,b,\xi}\;\; \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i
\quad\text{s.a.}\quad y_i(w^\top x_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0 .
$$

- **$C$ grande** → pocas violaciones permitidas → bajo sesgo, alta varianza
  (puede sobreajustar).
- **$C$ pequeño** → margen ancho, más violaciones toleradas → más regularizado.

### Forma de pérdida hinge
La SVM de margen blando equivale a minimizar la **pérdida hinge** regularizada:

$$
\min_{w,b}\; \tfrac{1}{2}\|w\|^2 + C\sum_i \max\big(0,\; 1 - y_i(w^\top x_i + b)\big).
$$
"""),
    ("md", r"""## 2. El problema dual y los vectores de soporte

Vía multiplicadores de Lagrange $\alpha_i \ge 0$, el **dual** es:

$$
\max_{\alpha}\; \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j\, y_i y_j\, (x_i^\top x_j)
\quad\text{s.a.}\quad 0 \le \alpha_i \le C,\;\; \sum_i \alpha_i y_i = 0 .
$$

Claves:
- Sólo los puntos con $\alpha_i > 0$ importan — son los **vectores de soporte**.
- Los datos entran **sólo por productos internos** $x_i^\top x_j$. Reemplazarlo
  por un **kernel** $K(x_i,x_j)$ nos permite trabajar en un espacio de alta
  dimensión *sin computar el mapeo* — el **truco del kernel**.
"""),
    ("md", r"""## 3. Kernels

Un kernel computa un producto interno en un espacio de características $\phi$
(posiblemente infinito-dimensional): $K(x,x') = \langle \phi(x), \phi(x') \rangle$.

| Kernel | Fórmula | Notas |
|---|---|---|
| Lineal | $K = x^\top x'$ | sin mapeo; rápido, datos esparsos de alta dim |
| Polinómico | $K = (\gamma\, x^\top x' + r)^d$ | interacciones de grado $d$ |
| RBF (gaussiano) | $K = \exp(-\gamma\|x-x'\|^2)$ | local, muy flexible; default |

Para el kernel **RBF**, $\gamma$ controla el alcance: $\gamma$ grande → bultos
estrechos → frontera ondulada (sobreajuste); $\gamma$ pequeño → frontera suave
(subajuste).
"""),
    ("code", BOOTSTRAP),
    ("code", r"""
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
setup_mlflow("module2-03-svm", backend="dagshub")

def plot_boundary(clf, X, y, ax, title):
    h = 0.02
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k", s=18)
    ax.set_title(title)
"""),
    ("code", r"""
# Comparamos kernels sobre make_moons
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (kernel, kw) in zip(axes, [
        ("linear", {}), ("poly", {"degree": 3}), ("rbf", {"gamma": 1.0})]):
    clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, **kw)).fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    plot_boundary(clf, X, y, ax, f"kernel {kernel} (acc test={acc:.2f})")
plt.tight_layout(); plt.show()
"""),
    ("md", r"""## 4. Efecto de $C$ y $\gamma$ (RBF)"""),
    ("code", r"""
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, C in enumerate([0.1, 1, 100]):
    for j, gamma in enumerate([0.1, 10]):
        ax = axes[j, i]
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="rbf", C=C, gamma=gamma)).fit(X_train, y_train)
        plot_boundary(clf, X, y, ax, f"C={C}, gamma={gamma}")
plt.suptitle("SVM RBF: C/gamma pequeños = suave, grandes = ondulado (sobreajuste)")
plt.tight_layout(); plt.show()
"""),
    ("md", r"""## 5. Ajustar $C$ y $\gamma$ con GridSearch + registrar en MLflow"""),
    ("code", r"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10]}
gs = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)

best = gs.best_estimator_
pred = best.predict(X_test)
metrics = {"cv_best_acc": gs.best_score_,
           "test_acc": accuracy_score(y_test, pred),
           "test_f1": f1_score(y_test, pred)}
params = {"C": gs.best_params_["svc__C"], "gamma": gs.best_params_["svc__gamma"], "kernel": "rbf"}
log_and_register(run_name="SVC-rbf-tuned", params=params, metrics=metrics,
                 model=best, flavor="sklearn",
                 registered_model_name="moons-svm",
                 tags={"algo": "SVC"})
print("Mejores params:", gs.best_params_, "| métricas:", metrics)
"""),
    ("md", r"""## 6. SVR — Regresión por vectores de soporte (bonus)

**Intuición:** la SVR ignora los errores pequeños. Usa un **tubo
$\epsilon$-insensible**: los errores menores que $\epsilon$ no penalizan, y aplica
la misma maquinaria de kernels. Objetivo:

$$
\min \tfrac12\|w\|^2 + C\sum_i (\xi_i + \xi_i^*)\quad
\text{s.a. } |y_i - (w^\top\phi(x_i)+b)| \le \epsilon + \xi.
$$
"""),
    ("code", r"""
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
rng = np.random.RandomState(0)
Xr = np.sort(5 * rng.rand(120, 1), axis=0)
yr = np.sin(Xr).ravel() + 0.1 * rng.randn(120)
svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, gamma=0.5, epsilon=0.1)).fit(Xr, yr)
xx = np.linspace(0, 5, 300).reshape(-1, 1)
plt.figure(figsize=(7,4))
plt.scatter(Xr, yr, s=12, label="datos")
plt.plot(xx, svr.predict(xx), "r", label="SVR (RBF)")
plt.legend(); plt.title(f"Ajuste SVR (RMSE train={np.sqrt(mean_squared_error(yr, svr.predict(Xr))):.3f})")
plt.tight_layout(); plt.show()
"""),
    ("md", r"""## 7. Resumen

- Las SVM maximizan el **margen**; el margen blando equilibra ancho de margen vs
  violaciones mediante $C$.
- El **dual** depende sólo de productos internos → el **truco del kernel** da
  fronteras no lineales de forma barata.
- **RBF** es un default fuerte; ajusta $C$ (regularización) y $\gamma$ (ancho del
  kernel) juntos. Escala siempre las variables primero.
"""),
]

# ===========================================================================
# 04 — NO SUPERVISADO
# ===========================================================================
nb04 = [
    ("md", r"""# 04 — Aprendizaje no supervisado (DBSCAN/OPTICS, GMM/KMeans, t-SNE/UMAP)

**Módulo 2 · Algoritmos avanzados de ML (+ MLflow)**

Vemos el **clustering por densidad** (DBSCAN vs OPTICS), el **clustering
probabilístico** (GMM vía EM vs KMeans) y la **reducción de dimensión para
visualización** (t-SNE vs UMAP). Registramos métricas (p. ej. silhouette) y
gráficas como artefactos de **MLflow**.
"""),
    ("code", BOOTSTRAP),
    ("code", r"""
setup_mlflow("module2-04-unsupervised", backend="dagshub")
"""),
    ("md", r"""## 1. DBSCAN — clustering por densidad

**Intuición:** DBSCAN agrupa puntos que están **densamente empaquetados** y marca
como **ruido** los puntos aislados. Dos hiperparámetros:
- `eps` ($\varepsilon$): radio del vecindario.
- `min_samples` (minPts): puntos necesarios dentro de $\varepsilon$ para ser
  *denso*.

Tipos de punto:
- **Núcleo:** tiene $\ge$ minPts vecinos dentro de $\varepsilon$.
- **Borde:** está dentro de $\varepsilon$ de un núcleo pero no es núcleo.
- **Ruido:** ninguno de los anteriores (etiqueta $-1$).

Los clústeres son conjuntos maximales de puntos *conectados por densidad*. A
diferencia de KMeans, DBSCAN encuentra **formas arbitrarias**, no necesita **$k$**
y es **robusto a outliers** — pero sufre cuando los clústeres tienen **densidades
muy distintas** (un único `eps` global no sirve para todos).
"""),
    ("code", r"""
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

Xm, _ = make_moons(n_samples=400, noise=0.06, random_state=42)
Xm = StandardScaler().fit_transform(Xm)
db = DBSCAN(eps=0.25, min_samples=5).fit(Xm)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int(np.sum(labels == -1))

plt.figure(figsize=(6,5))
plt.scatter(Xm[:,0], Xm[:,1], c=labels, cmap="tab10", s=18)
plt.title(f"DBSCAN sobre make_moons: {n_clusters} clústeres, {n_noise} puntos de ruido")
plt.tight_layout(); plt.show()
print("clústeres:", n_clusters, "ruido:", n_noise)
"""),
    ("md", r"""## 2. OPTICS — ordenar puntos para densidad variable

**Intuición:** OPTICS no se compromete con un único `eps`. Calcula para cada punto
una **distancia de alcanzabilidad** y produce un **ordenamiento**. El **gráfico de
alcanzabilidad** muestra valles = clústeres; valles profundos y estrechos son
densos, anchos y poco profundos son esparsos. Puedes extraer clústeres a *distintos*
niveles de densidad — así OPTICS maneja datos de **densidad variable** que DBSCAN
no puede con un solo `eps`.
"""),
    ("code", r"""
from sklearn.cluster import OPTICS
# Tres blobs con densidades muy distintas
Xa, _ = make_blobs(n_samples=[300, 100, 60], centers=[[0,0],[4,4],[8,0]],
                   cluster_std=[0.3, 0.8, 1.6], random_state=42)
opt = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05).fit(Xa)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
# Gráfico de alcanzabilidad
space = np.arange(len(Xa))
reach = opt.reachability_[opt.ordering_]
axes[0].plot(space, reach, lw=0.8)
axes[0].set_title("Gráfico de alcanzabilidad de OPTICS (valles = clústeres)")
axes[0].set_xlabel("puntos (orden del clúster)"); axes[0].set_ylabel("dist. alcanzabilidad")
# Clústeres en el espacio de características
axes[1].scatter(Xa[:,0], Xa[:,1], c=opt.labels_, cmap="tab10", s=15)
axes[1].set_title("Clústeres de OPTICS (densidad variable)")
plt.tight_layout(); plt.show()
"""),
    ("md", r"""## 3. KMeans vs modelos de mezcla gaussiana (GMM)

**KMeans** minimiza la distancia cuadrática dentro del clúster:
$$
\min_{\{S_k\}} \sum_{k=1}^{K} \sum_{x\in S_k} \|x-\mu_k\|^2 .
$$
Produce asignaciones **duras**, **esféricas** y de igual varianza.

**GMM** asume que los datos vienen de una mezcla de $K$ gaussianas:
$$
p(x) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(x \mid \mu_k, \Sigma_k),\qquad \sum_k \pi_k = 1.
$$
Se ajusta por **Esperanza–Maximización (EM)**:

- **Paso E:** calcular responsabilidades suaves (probabilidades posteriores)
$$
\gamma_{ik} = \frac{\pi_k\, \mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\, \mathcal{N}(x_i\mid\mu_j,\Sigma_j)} .
$$
- **Paso M:** actualizar los parámetros usando esos pesos
$$
\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}},\quad
\Sigma_k = \frac{\sum_i \gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top}{\sum_i \gamma_{ik}},\quad
\pi_k = \frac{1}{n}\sum_i \gamma_{ik}.
$$

GMM da asignaciones **suaves** y clústeres **elípticos** (covarianza completa), así
que ajusta clústeres estirados/correlacionados que KMeans falla.
"""),
    ("code", r"""
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Clústeres anisotrópicos (estirados) para exponer la debilidad de KMeans
Xg, yg = make_blobs(n_samples=600, centers=3, cluster_std=0.7, random_state=42)
transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
Xg = Xg @ transformation

km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xg)
gm = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(Xg)
km_lab, gm_lab = km.labels_, gm.predict(Xg)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(Xg[:,0], Xg[:,1], c=km_lab, cmap="tab10", s=12)
axes[0].set_title(f"KMeans (silhouette={silhouette_score(Xg, km_lab):.3f})")
axes[1].scatter(Xg[:,0], Xg[:,1], c=gm_lab, cmap="tab10", s=12)
axes[1].set_title(f"GMM (silhouette={silhouette_score(Xg, gm_lab):.3f})")
plt.tight_layout(); plt.show()
"""),
    ("code", r"""
# Registramos métricas de clustering + la comparación como artefactos de MLflow
import tempfile, os
with mlflow.start_run(run_name="kmeans-vs-gmm"):
    mlflow.log_params({"n_clusters": 3, "covariance_type": "full"})
    mlflow.log_metrics({"kmeans_silhouette": silhouette_score(Xg, km_lab),
                        "gmm_silhouette": silhouette_score(Xg, gm_lab),
                        "gmm_bic": gm.bic(Xg), "gmm_aic": gm.aic(Xg)})
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(Xg[:,0], Xg[:,1], c=gm_lab, cmap="tab10", s=12); ax.set_title("Clústeres GMM")
    tmp = os.path.join(tempfile.gettempdir(), "gmm_clusters.png")
    fig.savefig(tmp, dpi=110, bbox_inches="tight"); plt.close(fig)
    mlflow.log_artifact(tmp)
    print("Run de clustering + artefacto registrados:", tmp)
"""),
    ("md", r"""## 4. Reducción de dimensión para visualización: t-SNE vs UMAP

Ambos mapean datos de alta dimensión a 2-D para *visualizar* (no como features).

### t-SNE
Convierte distancias por pares en **probabilidades** (gaussiana en alta-D,
Student-$t$ en baja-D) y minimiza la **divergencia KL** entre ellas:
$$
C = \mathrm{KL}(P \,\|\, Q) = \sum_{i\neq j} p_{ij}\,\log\frac{p_{ij}}{q_{ij}}.
$$
Excelente revelando estructura **local** / clústeres, pero **lento**, estocástico
y las **distancias entre clústeres no son significativas** (mala estructura
global). Knob clave: `perplexity`.

### UMAP
Basado en teoría de variedades / **conjuntos simpliciales difusos**: modela los
datos como un grafo topológico difuso en alta-D y busca un layout en baja-D que lo
iguale. Es **mucho más rápido**, escala mejor y tiende a **preservar mejor la
estructura global** que t-SNE. Knobs clave: `n_neighbors` (local vs global) y
`min_dist` (compacidad de los clústeres).
"""),
    ("code", r"""
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
digits = load_digits()
Xd, yd = digits.data, digits.target
print("digits:", Xd.shape)

tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
Z_tsne = tsne.fit_transform(Xd)

try:
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    Z_umap = reducer.fit_transform(Xd)
    have_umap = True
except Exception as e:
    print("UMAP no disponible (", e, ") — usamos PCA en el panel derecho.")
    from sklearn.decomposition import PCA
    Z_umap = PCA(n_components=2).fit_transform(Xd)
    have_umap = False
"""),
    ("code", r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sc = axes[0].scatter(Z_tsne[:,0], Z_tsne[:,1], c=yd, cmap="tab10", s=8)
axes[0].set_title("t-SNE (divergencia KL, estructura local)")
axes[1].scatter(Z_umap[:,0], Z_umap[:,1], c=yd, cmap="tab10", s=8)
axes[1].set_title("UMAP (conjuntos simpliciales difusos)" if have_umap else "PCA (fallback de UMAP)")
fig.colorbar(sc, ax=axes, fraction=0.025, label="dígito")
plt.show()
"""),
    ("md", r"""## 5. Resumen

| Tarea | Método A | Método B | Cuándo gana B |
|---|---|---|---|
| Clustering por densidad | DBSCAN | OPTICS | densidades variables |
| Clustering por centroide | KMeans | GMM | clústeres elípticos / suaves |
| Visualización 2-D | t-SNE | UMAP | velocidad + estructura global |

La calidad del clustering es difícil de medir sin etiquetas — usa **silhouette**,
**BIC/AIC** (GMM) e *inspección del dominio*. Registra todo en MLflow para que las
gráficas y métricas queden adjuntas a cada run.
"""),
]

# ===========================================================================
# 05 — REDES NEURONALES (PyTorch)  — NOTEBOOK CLAVE
# ===========================================================================
nb05 = [
    ("md", r"""# 05 — Redes neuronales con PyTorch

**Módulo 2 · Algoritmos avanzados de ML (+ MLflow)**

Este es el notebook **más importante** del módulo. Construimos la intuición de una
red neuronal desde la **neurona simple** hasta un **perceptrón multicapa (MLP)**,
con **diagramas dibujados en código** (sin imágenes externas) y un balance entre
concepto y matemática. Recorremos:

1. La neurona simple → perceptrón → MLP.
2. Funciones de activación y sus derivadas.
3. El problema del **desvanecimiento del gradiente**.
4. **Decisiones de arquitectura ANTES de programar** (checklist).
5. **Forward pass y backpropagation** (regla de la cadena).
6. El algoritmo **SGD explicado con imágenes**.
7. **Regularización** (dropout, weight decay, early stopping, batchnorm).
8. **Implementación en PyTorch — dos casos**: regresión y clasificación,
   registrados en **MLflow**.

> La idea lleva: **el concepto primero, la fórmula compacta después, y los
> diagramas para iluminar**. Sin muros de ecuaciones.
"""),
    ("code", BOOTSTRAP),
    ("code", r"""
import torch
import torch.nn as nn
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
torch.manual_seed(42)
"""),
    # ---- 1. La neurona simple ----
    ("md", r"""## 1. La neurona simple

**Intuición:** una neurona artificial recibe varias entradas, las pondera, las
suma, le añade un *sesgo* (bias) y pasa el resultado por una **función de
activación** que decide cuánto "se enciende". Apilando muchas neuronas en capas
obtenemos una red capaz de aproximar funciones muy complejas.

Compactamente, para entradas $\mathbf{x}=(x_1,\dots,x_n)$, pesos $\mathbf{w}$ y
sesgo $b$:

$$
z = \mathbf{w}^\top \mathbf{x} + b, \qquad a = \phi(z).
$$

$z$ es la **pre-activación** (una combinación lineal) y $a$ la **activación** (la
salida tras la no linealidad $\phi$). Dibujemos esa neurona.
"""),
    ("code", r"""
# Diagrama de una neurona simple: x1..xn -> pesos -> suma -> +bias -> activación -> salida
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axis("off")

inputs = ["$x_1$", "$x_2$", "$x_3$", r"$\vdots$", "$x_n$"]
y_in = np.linspace(4.5, 0.5, len(inputs))
sum_xy = (5.2, 2.5)     # nodo de la suma
act_xy = (7.2, 2.5)     # nodo de activación
out_xy = (9.2, 2.5)     # salida

# Nodos de entrada y flechas ponderadas hacia la suma
for label, y in zip(inputs, y_in):
    ax.scatter(1.0, y, s=900, c="#cfe8ff", edgecolors="#1f77b4", zorder=3)
    ax.text(1.0, y, label, ha="center", va="center", fontsize=12)
    if label != r"$\vdots$":
        ax.annotate("", xy=sum_xy, xytext=(1.4, y),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.text((1.4 + sum_xy[0]) / 2, (y + sum_xy[1]) / 2 + 0.12,
                r"$w$", color="#d62728", fontsize=10)

# Nodo suma (Σ) con el bias entrando
ax.scatter(*sum_xy, s=1500, c="#fff2cc", edgecolors="#b8860b", zorder=3)
ax.text(*sum_xy, r"$\sum$", ha="center", va="center", fontsize=16)
ax.scatter(sum_xy[0], sum_xy[1] + 2.0, s=700, c="#e2f0d9", edgecolors="green", zorder=3)
ax.text(sum_xy[0], sum_xy[1] + 2.0, "$b$", ha="center", va="center", fontsize=12)
ax.annotate("", xy=sum_xy, xytext=(sum_xy[0], sum_xy[1] + 1.6),
            arrowprops=dict(arrowstyle="->", color="green"))

# Suma -> activación
ax.annotate("", xy=act_xy, xytext=(sum_xy[0] + 0.4, sum_xy[1]),
            arrowprops=dict(arrowstyle="->", color="black"))
ax.text((sum_xy[0] + act_xy[0]) / 2, sum_xy[1] + 0.25, "$z$", fontsize=12)
ax.scatter(*act_xy, s=1500, c="#f4cccc", edgecolors="#cc0000", zorder=3)
ax.text(*act_xy, r"$\phi$", ha="center", va="center", fontsize=15)

# Activación -> salida
ax.annotate("", xy=out_xy, xytext=(act_xy[0] + 0.4, act_xy[1]),
            arrowprops=dict(arrowstyle="->", color="black"))
ax.text((act_xy[0] + out_xy[0]) / 2, act_xy[1] + 0.25, "$a$", fontsize=12)
ax.text(out_xy[0], out_xy[1], "salida", ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round", fc="#ddd"))

ax.text(5.2, 5.4, r"$z = \mathbf{w}^\top \mathbf{x} + b \quad\to\quad a = \phi(z)$",
        ha="center", fontsize=14)
ax.set_xlim(0, 10.2); ax.set_ylim(-0.3, 6)
plt.title("La neurona simple", fontsize=13)
plt.tight_layout(); plt.show()
"""),
    ("md", r"""### Del perceptrón al MLP

**Intuición:** una sola neurona (el *perceptrón*) sólo traza una **frontera
lineal**. Apilando capas de neuronas con activaciones no lineales, cada capa
transforma el espacio y la siguiente combina esas transformaciones, hasta que la
red puede separar datos que ninguna recta podría. A esto lo llamamos **perceptrón
multicapa (MLP)**: una capa de entrada, una o más **capas ocultas** y una capa de
salida.
"""),
    ("code", r"""
# Diagrama de una pequeña red por capas (MLP): 3 entradas -> 4 ocultas -> 4 ocultas -> 1 salida
def draw_layer(ax, x, n, color, prefix):
    ys = np.linspace(0.5, 4.5, n)
    coords = []
    for i, y in enumerate(ys):
        ax.scatter(x, y, s=600, c=color, edgecolors="k", zorder=3)
        coords.append((x, y))
    return coords

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axis("off")
layers = [draw_layer(ax, 1, 3, "#cfe8ff", "in"),
          draw_layer(ax, 4, 4, "#fff2cc", "h1"),
          draw_layer(ax, 7, 4, "#fff2cc", "h2"),
          draw_layer(ax, 10, 1, "#f4cccc", "out")]
# Conexiones totalmente conectadas entre capas consecutivas
for a_layer, b_layer in zip(layers[:-1], layers[1:]):
    for (xa, ya) in a_layer:
        for (xb, yb) in b_layer:
            ax.plot([xa, xb], [ya, yb], color="gray", lw=0.4, zorder=1)

names = ["entrada\n(3)", "oculta 1\n(4)", "oculta 2\n(4)", "salida\n(1)"]
for x, name in zip([1, 4, 7, 10], names):
    ax.text(x, 5.1, name, ha="center", fontsize=10)
ax.set_xlim(0, 11); ax.set_ylim(0, 5.6)
plt.title("Perceptrón multicapa (MLP): capas totalmente conectadas", fontsize=13)
plt.tight_layout(); plt.show()
"""),
    # ---- 2. Funciones de activación ----
    ("md", r"""## 2. Funciones de activación

**Intuición:** sin no linealidad, apilar capas equivale a una sola capa lineal.
La activación es lo que da poder expresivo a la red. Estas son las más usadas:

| Nombre | Fórmula | Rango | Nota |
|---|---|---|---|
| Sigmoid | $\sigma(z)=\frac{1}{1+e^{-z}}$ | $(0,1)$ | satura → gradientes que se desvanecen |
| Tanh | $\tanh(z)$ | $(-1,1)$ | centrada en cero |
| ReLU | $\max(0,z)$ | $[0,\infty)$ | barata, esparsa; neuronas "muertas" |
| Leaky ReLU | $\max(\alpha z, z)$ | $\mathbb{R}$ | evita neuronas muertas |
| GELU | $z\,\Phi(z)$ | $\mathbb{R}$ | ReLU suave; usada en Transformers |

Dibujemos cada activación **y su derivada** (clave para entender el gradiente).
"""),
    ("code", r"""
import torch.nn.functional as F

z = torch.linspace(-6, 6, 400, requires_grad=True)

def deriv(fn, z):
    z = z.clone().detach().requires_grad_(True)
    y = fn(z)
    y.sum().backward()
    return y.detach(), z.grad.detach()

acts = {
    "sigmoid":    lambda t: torch.sigmoid(t),
    "tanh":       lambda t: torch.tanh(t),
    "relu":       lambda t: F.relu(t),
    "leaky_relu": lambda t: F.leaky_relu(t, negative_slope=0.1),
    "gelu":       lambda t: F.gelu(t),
}

fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
zz = z.detach().numpy()
for j, (name, fn) in enumerate(acts.items()):
    y, g = deriv(fn, z)
    axes[0, j].plot(zz, y.numpy(), color="#1f77b4")
    axes[0, j].set_title(name); axes[0, j].axhline(0, color="k", lw=0.4)
    axes[0, j].axvline(0, color="k", lw=0.4)
    axes[1, j].plot(zz, g.numpy(), color="#d62728")
    axes[1, j].set_title(f"{name}'  (derivada)"); axes[1, j].axhline(0, color="k", lw=0.4)
    axes[1, j].axvline(0, color="k", lw=0.4)
axes[0, 0].set_ylabel("activación  $\\phi(z)$")
axes[1, 0].set_ylabel("derivada  $\\phi'(z)$")
plt.suptitle("Funciones de activación (arriba) y sus derivadas (abajo)", fontsize=13)
plt.tight_layout(); plt.show()
"""),
    ("md", r"""**Pros y contras (resumen práctico):**

| Activación | Pros | Contras |
|---|---|---|
| Sigmoid | salida en $(0,1)$, útil como probabilidad final | satura, derivada máx. $0.25$ → desvanecimiento |
| Tanh | centrada en cero (mejor que sigmoid en capas ocultas) | sigue saturando en los extremos |
| ReLU | rápida, no satura para $z>0$, induce esparsidad | "neuronas muertas" si $z<0$ siempre |
| Leaky ReLU | arregla las neuronas muertas (pendiente pequeña en $z<0$) | un hiperparámetro extra ($\alpha$) |
| GELU | suave, muy buen rendimiento en redes profundas | algo más cara de computar |

**Regla de oro:** usa **ReLU** (o GELU en redes grandes) en las capas ocultas, y
reserva **sigmoid/softmax** para la **salida** de clasificación.
"""),
    # ---- 3. Vanishing gradient ----
    ("md", r"""## 3. El problema del desvanecimiento del gradiente

**Intuición:** para entrenar, el gradiente del error viaja *hacia atrás* capa por
capa, **multiplicándose** por la derivada de la activación en cada paso. Si esas
derivadas son pequeñas (como en sigmoid/tanh, que saturan), el producto se hace
**diminuto** y las primeras capas casi no aprenden. Eso es el *desvanecimiento del
gradiente*.

El dato clave: la derivada de la sigmoid nunca supera **0.25**. Encadenando $L$
capas, el gradiente se escala aproximadamente por $(0.25)^L$ — que cae a cero muy
rápido. Veámoslo.
"""),
    ("code", r"""
# (a) Saturación: la derivada de sigmoid/tanh tiende a 0 en los extremos.
zz = torch.linspace(-8, 8, 400)
sig = torch.sigmoid(zz); dsig = sig * (1 - sig)
th = torch.tanh(zz); dth = 1 - th**2

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(zz, dsig, label="sigmoid'  (máx 0.25)", color="#1f77b4")
axes[0].plot(zz, dth, label="tanh'  (máx 1.0)", color="#ff7f0e")
axes[0].axhline(0.25, color="#1f77b4", ls="--", lw=0.8)
axes[0].set_title("Derivadas que saturan → casi 0 en los extremos")
axes[0].set_xlabel("z"); axes[0].set_ylabel("derivada"); axes[0].legend()

# (b) Cómo encoge la magnitud del gradiente al atravesar muchas capas.
layers = np.arange(1, 21)
for dmax, name, c in [(0.25, "sigmoid (deriv≈0.25)", "#1f77b4"),
                      (1.0, "tanh (deriv≈1.0, mejor caso)", "#ff7f0e"),
                      (1.0, "ReLU (deriv=1 para z>0)", "#2ca02c")]:
    axes[1].plot(layers, dmax ** layers, marker="o", ms=3, label=name, color=c)
axes[1].set_yscale("log")
axes[1].set_title("Magnitud del gradiente ≈ (derivada)$^{n}$,  n = # de capas")
axes[1].set_xlabel("# de capas atravesadas"); axes[1].set_ylabel("factor del gradiente (log)")
axes[1].legend()
plt.tight_layout(); plt.show()
"""),
    ("md", r"""**Remedios habituales:**

- **ReLU (y variantes):** su derivada es $1$ para $z>0$, así que no encoge el
  gradiente — el remedio más simple y efectivo.
- **BatchNorm:** normaliza las pre-activaciones por mini-batch, manteniéndolas en
  una zona donde las derivadas no saturan.
- **Conexiones residuales (skip connections):** crean un "atajo" $x + f(x)$ por el
  que el gradiente fluye directo, base de las redes muy profundas (ResNet).
- **Inicialización cuidada** (Xavier/He): fija la escala inicial de los pesos para
  que las señales no se atenúen ni exploten al propagarse.
"""),
    # ---- 4. Decisiones de arquitectura ----
    ("md", r"""## 4. Decisiones de arquitectura ANTES de programar

Antes de escribir una sola línea de PyTorch conviene decidir la arquitectura. Esta
es una **checklist práctica**:

| Decisión | Pregunta guía | Default razonable |
|---|---|---|
| **# de capas** (profundidad) | ¿problema simple o complejo? | empieza con 1–3 ocultas |
| **# de neuronas** (ancho) | ¿cuánta capacidad necesito? | 32–256 por capa; baja hacia la salida |
| **Activación oculta** | ¿qué evita el desvanecimiento? | **ReLU** (o GELU) |
| **Capa de SALIDA** | ¿qué predigo? | **regresión → lineal** (sin activación); **clasif. binaria → 1 neurona + sigmoid**; **multiclase → K neuronas + softmax** |
| **Función de PÉRDIDA** | acorde a la salida | **regresión → MSE**; **binaria → BCE**; **multiclase → cross-entropy** |
| **Optimizador** | ¿robusto por defecto? | **Adam** (o SGD+momentum si quieres exprimir) |
| **Learning rate** | el hiperparámetro más sensible | $10^{-3}$ con Adam; ajústalo primero |
| **Batch size** | ¿memoria vs estabilidad? | 32–256 |
| **Épocas** | ¿cuándo parar? | muchas + **early stopping** |
| **Regularización** | ¿señales de sobreajuste? | dropout, weight decay, batchnorm |

> **Par output ↔ pérdida (lo más importante de recordar):**
> - **Regresión:** salida **lineal** + **MSE**.
> - **Clasificación binaria:** salida con **1 logit** + **BCEWithLogits**.
> - **Clasificación multiclase:** **K logits** + **CrossEntropy**.
"""),
    # ---- 5. Forward + backprop ----
    ("md", r"""## 5. Forward pass y backpropagation

**Forward pass (intuición):** los datos entran y avanzan capa por capa. Para la
capa $\ell$:

$$
z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)},\qquad
a^{(\ell)} = \phi\big(z^{(\ell)}\big),\qquad a^{(0)} = x.
$$

**Backpropagation (intuición):** una vez calculada la pérdida, queremos saber
*cuánto contribuye cada peso al error*. La **regla de la cadena** nos deja
calcular ese gradiente eficientemente propagándolo **hacia atrás**. Definiendo el
error por capa $\delta^{(\ell)} = \partial L / \partial z^{(\ell)}$:

$$
\delta^{(L)} = \nabla_a L \odot \phi'(z^{(L)}),\qquad
\delta^{(\ell)} = \big( W^{(\ell+1)\top} \delta^{(\ell+1)} \big) \odot \phi'(z^{(\ell)}),
$$

y los gradientes que actualizan los parámetros:

$$
\frac{\partial L}{\partial W^{(\ell)}} = \delta^{(\ell)} a^{(\ell-1)\top},\qquad
\frac{\partial L}{\partial b^{(\ell)}} = \delta^{(\ell)} .
$$

El **autograd** de PyTorch construye el grafo de cómputo y hace todo esto
automáticamente al llamar `loss.backward()`.
"""),
    ("code", r"""
# Diagrama del flujo de gradiente: forward (→) y backward (←) por una red pequeña.
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.axis("off")
boxes = ["$x$", "capa 1\n$z^{(1)}, a^{(1)}$", "capa 2\n$z^{(2)}, a^{(2)}$",
         "salida\n$\\hat y$", "pérdida\n$L$"]
xs = np.linspace(0.5, 9.5, len(boxes))
for x, b in zip(xs, boxes):
    ax.text(x, 1.5, b, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="#cfe8ff", ec="#1f77b4"))
# forward (arriba, azul)
for xa, xb in zip(xs[:-1], xs[1:]):
    ax.annotate("", xy=(xb - 0.6, 1.85), xytext=(xa + 0.6, 1.85),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.6))
ax.text(xs.mean(), 2.5, "forward  →  (predicción)", ha="center", color="#1f77b4", fontsize=11)
# backward (abajo, rojo)
for xa, xb in zip(xs[:-1], xs[1:]):
    ax.annotate("", xy=(xa + 0.6, 1.15), xytext=(xb - 0.6, 1.15),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
ax.text(xs.mean(), 0.45, r"backward  ←  (gradientes vía regla de la cadena)",
        ha="center", color="#d62728", fontsize=11)
ax.set_xlim(0, 10); ax.set_ylim(0, 3)
plt.title("Forward pass y backpropagation", fontsize=13)
plt.tight_layout(); plt.show()
"""),
    # ---- 6. SGD con imágenes ----
    ("md", r"""## 6. El algoritmo SGD explicado con imágenes

**Intuición:** entrenar es *bajar una montaña* (la superficie de pérdida) dando
pasos en la dirección de máxima pendiente descendente, que es el **gradiente
negativo**. La regla de actualización del descenso de gradiente es:

$$
\theta \leftarrow \theta - \eta\,\nabla_\theta L,
$$

donde $\eta$ es la **tasa de aprendizaje** (el tamaño del paso). El "estocástico"
(SGD) viene de estimar el gradiente con un **mini-batch** en vez de todos los
datos. Veamos tres imágenes.
"""),
    ("code", r"""
# (a) Superficie de pérdida 2D con el camino del descenso de gradiente (quiver).
def loss(w):  # cuenco elíptico simple
    return 0.5 * (w[0]**2 / 3.0 + w[1]**2)
def grad(w):
    return np.array([w[0] / 3.0, w[1] * 2.0])

w = np.array([5.0, 4.0]); eta = 0.25; path = [w.copy()]
for _ in range(18):
    w = w - eta * grad(w); path.append(w.copy())
path = np.array(path)

gx, gy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-5, 5, 200))
gz = 0.5 * (gx**2 / 3.0 + gy**2)
plt.figure(figsize=(7, 5.5))
cs = plt.contour(gx, gy, gz, levels=20, cmap="viridis")
plt.clabel(cs, inline=True, fontsize=7)
plt.quiver(path[:-1, 0], path[:-1, 1],
           path[1:, 0] - path[:-1, 0], path[1:, 1] - path[:-1, 1],
           angles="xy", scale_units="xy", scale=1, color="red", width=0.005)
plt.scatter(*path[0], c="red", s=60, label="inicio")
plt.scatter(0, 0, c="black", marker="*", s=160, label="mínimo")
plt.title("Descenso de gradiente sobre la superficie de pérdida")
plt.xlabel(r"$\theta_1$"); plt.ylabel(r"$\theta_2$"); plt.legend()
plt.tight_layout(); plt.show()
"""),
    ("code", r"""
# (b) Efecto de la learning rate sobre una parábola 1D: muy pequeña / buena / muy grande.
def f(x): return x**2
def df(x): return 2*x
xs = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (eta, title) in zip(axes, [(0.05, "muy pequeña (lenta)"),
                                   (0.4, "buena (converge)"),
                                   (1.02, "muy grande (diverge)")]):
    ax.plot(xs, f(xs), color="#999")
    x = 4.5
    for _ in range(12):
        ax.scatter(x, f(x), color="red", s=25, zorder=3)
        x_new = x - eta * df(x)
        ax.annotate("", xy=(x_new, f(x_new)), xytext=(x, f(x)),
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
        x = x_new
        if abs(x) > 6:  # se escapó
            break
    ax.set_title(f"η = {eta}\n{title}"); ax.set_xlabel("θ"); ax.set_ylabel("L(θ)")
plt.suptitle("Efecto de la tasa de aprendizaje", fontsize=13)
plt.tight_layout(); plt.show()
"""),
    ("code", r"""
# (c) Batch vs mini-batch vs estocástico: cómo de "ruidoso" es el camino al mínimo.
np.random.seed(0)
def noisy_path(noise):
    w = np.array([5.0, 4.0]); pts = [w.copy()]
    for _ in range(40):
        g = grad(w) + np.random.randn(2) * noise
        w = w - 0.18 * g; pts.append(w.copy())
    return np.array(pts)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.contour(gx, gy, gz, levels=18, cmap="Greys", linewidths=0.5)
for noise, name, c in [(0.0, "batch completo (suave)", "#1f77b4"),
                       (0.6, "mini-batch (algo de ruido)", "#ff7f0e"),
                       (1.8, "estocástico (muy ruidoso)", "#2ca02c")]:
    p = noisy_path(noise)
    ax.plot(p[:, 0], p[:, 1], marker="o", ms=2, label=name, color=c, alpha=0.8)
ax.scatter(0, 0, c="black", marker="*", s=160)
ax.set_title("Batch vs mini-batch vs estocástico")
ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$"); ax.legend()
plt.tight_layout(); plt.show()
"""),
    ("md", r"""**Más allá del SGD básico:** dos mejoras muy usadas.

- **Momentum:** acumula una *velocidad* para amortiguar oscilaciones y acelerar en
  direcciones consistentes:
$$
v_t = \mu v_{t-1} + g_t,\qquad \theta_{t+1} = \theta_t - \eta\, v_t .
$$
- **Adam:** tasa de aprendizaje *adaptativa por parámetro*, combinando estimaciones
  del 1er y 2º momento del gradiente. Es un default robusto:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\quad
\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$
"""),
    # ---- 7. Regularización ----
    ("md", r"""## 7. Regularización en redes neuronales

Las redes tienen muchos parámetros y sobreajustan con facilidad. Cuatro técnicas
clave y **cómo aparecen en PyTorch**:

- **Dropout:** apaga al azar una fracción $p$ de activaciones en entrenamiento;
  evita la co-adaptación y actúa como un ensemble.
  `nn.Dropout(p=0.3)`.
- **Weight decay (L2):** añade $\frac{\lambda}{2}\|\theta\|^2$ a la pérdida,
  encogiendo los pesos.
  `torch.optim.Adam(..., weight_decay=1e-4)`.
- **Early stopping:** detiene el entrenamiento cuando la pérdida de validación deja
  de mejorar. Se implementa con un contador de "paciencia" en el bucle.
- **BatchNorm:** normaliza las pre-activaciones por mini-batch, estabilizando y
  acelerando el entrenamiento (y mitigando el desvanecimiento).
  `nn.BatchNorm1d(num_features)`.

A continuación las usamos todas en los dos casos prácticos.
"""),
    # ---- 8. PyTorch — utilidades comunes ----
    ("md", r"""## 8. Implementación en PyTorch — dos casos

Construimos un **bucle de entrenamiento genérico** (con validación y early
stopping) y lo reutilizamos en:

- **Caso A — Regresión:** California housing, MLP con **salida lineal + MSE**.
- **Caso B — Clasificación:** breast cancer, MLP con **sigmoid + BCE**.

Ambos registran params/métricas/modelo en **MLflow** y los registran en el
registry vía los helpers existentes.
"""),
    ("code", r"""
from torch.utils.data import TensorDataset, DataLoader

def make_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te, batch=32, target_2d=True):
    # Crea DataLoaders de train/val/test a partir de arrays numpy.
    def to_ds(Xa, ya):
        yt = torch.tensor(ya, dtype=torch.float32)
        if target_2d:
            yt = yt.unsqueeze(1)
        return TensorDataset(torch.tensor(Xa, dtype=torch.float32), yt)
    return (DataLoader(to_ds(X_tr, y_tr), batch_size=batch, shuffle=True),
            DataLoader(to_ds(X_val, y_val), batch_size=256, shuffle=False),
            DataLoader(to_ds(X_te, y_te), batch_size=256, shuffle=False))


class MLP(nn.Module):
    # MLP genérico con BatchNorm + ReLU + Dropout. La capa de salida (out_dim,
    # sin activación) la decide la tarea: 1 logit para regresión o clasif. binaria.
    def __init__(self, in_dim, hidden=(64, 32), out_dim=1, p_drop=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, out_dim)]   # salida lineal (logits / valor)
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
"""),
    ("code", r"""
def train_loop(model, train_dl, val_dl, criterion, optimizer, epochs, patience,
               eval_metric):
    # Bucle de entrenamiento genérico con validación y early stopping.
    # eval_metric(model, dl) -> dict con métricas extra por época (ej. RMSE, accuracy).
    # Devuelve (history, best_state). history["val_metric"] es una lista de dicts
    # (una entrada por época) que luego graficamos y enviamos a MLflow.
    history = {"train_loss": [], "val_loss": [], "val_metric": []}
    best_val, best_state, wait = float("inf"), None, 0

    def epoch_pass(dl, train):
        model.train() if train else model.eval()
        tot, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for xb, yb in dl:
                if train:
                    optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                if train:
                    loss.backward(); optimizer.step()
                tot += loss.item() * len(xb); n += len(xb)
        return tot / n

    for epoch in range(1, epochs + 1):
        tr = epoch_pass(train_dl, True)
        va = epoch_pass(val_dl, False)
        history["train_loss"].append(tr); history["val_loss"].append(va)
        if eval_metric is not None:
            history["val_metric"].append(eval_metric(model, val_dl))
        # early stopping sobre la pérdida de validación
        if va < best_val - 1e-4:
            best_val = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping en la época {epoch}")
                break
        if epoch % 10 == 0:
            print(f"época {epoch:3d} | train {tr:.4f} | val {va:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_state
"""),
    # ---- 8A. Regresión ----
    ("md", r"""### Caso A — Regresión (California housing)

Salida **lineal** (un valor continuo) + pérdida **MSE**. Reportamos el **RMSE** en
test y registramos todo en MLflow.
"""),
    ("code", r"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

setup_mlflow("module2-05-neural-networks", backend="dagshub")

reg = fetch_california_housing()
Xr, yr = reg.data.astype("float32"), reg.target.astype("float32")
Xr_tr, Xr_tmp, yr_tr, yr_tmp = train_test_split(Xr, yr, test_size=0.3, random_state=42)
Xr_val, Xr_te, yr_val, yr_te = train_test_split(Xr_tmp, yr_tmp, test_size=0.5, random_state=42)

# Estandarizamos features (y dejamos el target tal cual, en cientos de miles de $).
xsc = StandardScaler().fit(Xr_tr)
Xr_tr, Xr_val, Xr_te = (xsc.transform(a).astype("float32") for a in (Xr_tr, Xr_val, Xr_te))

reg_train_dl, reg_val_dl, reg_test_dl = make_loaders(
    Xr_tr, yr_tr, Xr_val, yr_val, Xr_te, yr_te, batch=64, target_2d=True)
print("Regresión — train/val/test:", len(Xr_tr), len(Xr_val), len(Xr_te),
      "| features:", Xr.shape[1])
"""),
    ("code", r"""
HP_REG = {"task": "regression", "lr": 1e-3, "weight_decay": 1e-4, "epochs": 100,
          "batch_size": 64, "hidden": "64,32", "dropout": 0.2,
          "optimizer": "adam", "patience": 12, "output": "linear", "loss": "MSE"}

torch.manual_seed(42)
reg_model = MLP(Xr.shape[1], hidden=(64, 32), out_dim=1, p_drop=0.2)
reg_criterion = nn.MSELoss()
reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=HP_REG["lr"],
                                 weight_decay=HP_REG["weight_decay"])

def rmse_on(model, dl):
    model.eval(); se, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            pred = model(xb)
            se += ((pred - yb) ** 2).sum().item(); n += len(xb)
    return float(np.sqrt(se / n))

with mlflow.start_run(run_name="mlp-california-regression") as run:
    mlflow.log_params(HP_REG)
    hist_reg, _ = train_loop(reg_model, reg_train_dl, reg_val_dl, reg_criterion,
                             reg_optimizer, HP_REG["epochs"], HP_REG["patience"],
                             eval_metric=lambda m, dl: {"val_rmse": rmse_on(m, dl)})
    # Métricas por época -> MLflow en UN solo request (log_batch); con un
    # servidor remoto (DagsHub) esto evita ~una llamada HTTP por época.
    from mlflow.entities import Metric
    import time as _time
    _ts = int(_time.time() * 1000)
    epoch_metrics = []
    for ep, (tl, vl) in enumerate(zip(hist_reg["train_loss"], hist_reg["val_loss"]), 1):
        epoch_metrics += [Metric("train_mse", tl, _ts, ep),
                          Metric("val_mse", vl, _ts, ep),
                          Metric("val_rmse", hist_reg["val_metric"][ep - 1]["val_rmse"], _ts, ep)]
    mlflow.MlflowClient().log_batch(run.info.run_id, metrics=epoch_metrics)
    test_rmse = rmse_on(reg_model, reg_test_dl)
    mlflow.log_metric("test_rmse", test_rmse)
    # Artefactos extra: arquitectura del modelo como texto
    mlflow.log_text(str(reg_model), "model_summary.txt")
    # Log + registro EN UN SOLO PASO: registered_model_name= es la vía
    # compatible con MLflow 2/3 y DagsHub. Registrar después con una URI
    # "runs:/<id>/model" falla en MLflow 3 (los modelos son *logged models*,
    # no artefactos del run).
    reg_name = "california-housing-mlp" if registry_available() else None
    try:
        mlflow.pytorch.log_model(reg_model, name="model",
                                 input_example=Xr_te[:5],
                                 registered_model_name=reg_name)
    except TypeError:
        mlflow.pytorch.log_model(reg_model, "model",
                                 input_example=Xr_te[:5],
                                 registered_model_name=reg_name)
    print(f"Modelo registrado: '{reg_name}'" if reg_name
          else "Registry no disponible (file store) — registro omitido.")
    print(f"TEST regresión — RMSE: {test_rmse:.4f}")
"""),
    ("code", r"""
# Curvas por época (pérdida + RMSE de validación) y las ENVIAMOS a MLflow como
# artefacto con mlflow.log_figure (reabrimos el mismo run con su run_id).
val_rmse_curve = [m["val_rmse"] for m in hist_reg["val_metric"]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(hist_reg["train_loss"], label="train")
axes[0].plot(hist_reg["val_loss"], label="val")
axes[0].set_title("Regresión — pérdida (MSE) por época")
axes[0].set_xlabel("época"); axes[0].set_ylabel("MSE"); axes[0].legend()
axes[1].plot(val_rmse_curve, color="green")
axes[1].set_title("Regresión — RMSE de validación por época")
axes[1].set_xlabel("época"); axes[1].set_ylabel("RMSE")
fig.tight_layout()

with mlflow.start_run(run_id=run.info.run_id):
    mlflow.log_figure(fig, "plots/curvas_regresion.png")
print("Figura de curvas enviada a MLflow: plots/curvas_regresion.png")
plt.show()
"""),
    # ---- 8B. Clasificación ----
    ("md", r"""### Caso B — Clasificación (breast cancer)

Salida con **1 logit + sigmoid** + pérdida **BCE** (usamos `BCEWithLogitsLoss`,
numéricamente estable, que aplica la sigmoid internamente). Reportamos **accuracy**
y **ROC-AUC** y registramos en MLflow.
"""),
    ("code", r"""
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, accuracy_score

clf = load_breast_cancer()
Xc, yc = clf.data.astype("float32"), clf.target.astype("float32")
Xc_tr, Xc_tmp, yc_tr, yc_tmp = train_test_split(
    Xc, yc, test_size=0.3, random_state=42, stratify=yc)
Xc_val, Xc_te, yc_val, yc_te = train_test_split(
    Xc_tmp, yc_tmp, test_size=0.5, random_state=42, stratify=yc_tmp)

csc = StandardScaler().fit(Xc_tr)
Xc_tr, Xc_val, Xc_te = (csc.transform(a).astype("float32") for a in (Xc_tr, Xc_val, Xc_te))

clf_train_dl, clf_val_dl, clf_test_dl = make_loaders(
    Xc_tr, yc_tr, Xc_val, yc_val, Xc_te, yc_te, batch=32, target_2d=True)
print("Clasificación — train/val/test:", len(Xc_tr), len(Xc_val), len(Xc_te),
      "| features:", Xc.shape[1])
"""),
    ("code", r"""
HP_CLF = {"task": "classification", "lr": 1e-3, "weight_decay": 1e-4, "epochs": 100,
          "batch_size": 32, "hidden": "64,32", "dropout": 0.3,
          "optimizer": "adam", "patience": 12, "output": "sigmoid", "loss": "BCE"}

torch.manual_seed(42)
clf_model = MLP(Xc.shape[1], hidden=(64, 32), out_dim=1, p_drop=0.3)
clf_criterion = nn.BCEWithLogitsLoss()   # aplica sigmoid internamente
clf_optimizer = torch.optim.Adam(clf_model.parameters(), lr=HP_CLF["lr"],
                                 weight_decay=HP_CLF["weight_decay"])

def clf_eval(model, dl):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for xb, yb in dl:
            prob = torch.sigmoid(model(xb))
            ps.append(prob.numpy()); ys.append(yb.numpy())
    y = np.vstack(ys).ravel(); p = np.vstack(ps).ravel()
    acc = accuracy_score(y, (p >= 0.5).astype(int))
    auc = roc_auc_score(y, p)
    return {"accuracy": float(acc), "roc_auc": float(auc)}

with mlflow.start_run(run_name="mlp-breast-cancer-clf") as run:
    mlflow.log_params(HP_CLF)
    hist_clf, _ = train_loop(clf_model, clf_train_dl, clf_val_dl, clf_criterion,
                             clf_optimizer, HP_CLF["epochs"], HP_CLF["patience"],
                             eval_metric=clf_eval)
    # Métricas por época -> MLflow en UN solo request (log_batch)
    from mlflow.entities import Metric
    import time as _time
    _ts = int(_time.time() * 1000)
    epoch_metrics = []
    for ep, (tl, vl) in enumerate(zip(hist_clf["train_loss"], hist_clf["val_loss"]), 1):
        m = hist_clf["val_metric"][ep - 1]
        epoch_metrics += [Metric("train_bce", tl, _ts, ep),
                          Metric("val_bce", vl, _ts, ep),
                          Metric("val_accuracy", m["accuracy"], _ts, ep),
                          Metric("val_roc_auc", m["roc_auc"], _ts, ep)]
    mlflow.MlflowClient().log_batch(run.info.run_id, metrics=epoch_metrics)
    test_metrics = clf_eval(clf_model, clf_test_dl)
    mlflow.log_metrics({"test_accuracy": test_metrics["accuracy"],
                        "test_roc_auc": test_metrics["roc_auc"]})
    # Artefactos extra: arquitectura del modelo como texto
    mlflow.log_text(str(clf_model), "model_summary.txt")
    # Log + registro en un solo paso (compatible MLflow 2/3 y DagsHub) — ver
    # el comentario del caso de regresión.
    reg_name = "breast-cancer-mlp" if registry_available() else None
    try:
        mlflow.pytorch.log_model(clf_model, name="model",
                                 input_example=Xc_te[:5],
                                 registered_model_name=reg_name)
    except TypeError:
        mlflow.pytorch.log_model(clf_model, "model",
                                 input_example=Xc_te[:5],
                                 registered_model_name=reg_name)
    print(f"Modelo registrado: '{reg_name}'" if reg_name
          else "Registry no disponible (file store) — registro omitido.")
    print(f"TEST clasificación — acc: {test_metrics['accuracy']:.4f} | "
          f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
"""),
    ("code", r"""
# Curvas por época (pérdida BCE + accuracy/ROC-AUC) enviadas a MLflow.
val_acc_curve = [m["accuracy"] for m in hist_clf["val_metric"]]
val_auc_curve = [m["roc_auc"] for m in hist_clf["val_metric"]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(hist_clf["train_loss"], label="train")
axes[0].plot(hist_clf["val_loss"], label="val")
axes[0].set_title("Clasificación — pérdida (BCE) por época")
axes[0].set_xlabel("época"); axes[0].set_ylabel("BCE"); axes[0].legend()
axes[1].plot(val_acc_curve, label="accuracy")
axes[1].plot(val_auc_curve, label="ROC-AUC")
axes[1].set_title("Clasificación — métricas de validación por época")
axes[1].set_xlabel("época"); axes[1].set_ylabel("score"); axes[1].legend()
fig.tight_layout()

with mlflow.start_run(run_id=run.info.run_id):
    mlflow.log_figure(fig, "plots/curvas_clasificacion.png")
print("Figura de curvas enviada a MLflow: plots/curvas_clasificacion.png")
plt.show()
"""),
    ("md", r"""### Diagrama de la arquitectura del MLP final

Dibujamos (en código, como el resto de diagramas del notebook) la arquitectura
del MLP final de clasificación: **30 features → Linear(30→64) → Linear(64→32) →
Linear(32→1)**, cada capa oculta con **BatchNorm + ReLU + Dropout**, y un único
**logit** de salida que la sigmoid (dentro de `BCEWithLogitsLoss`) convierte en
probabilidad. El diagrama se guarda también como **artefacto del run** en MLflow.
"""),
    ("code", r"""
def draw_mlp_architecture(layer_sizes, layer_labels, title, max_neurons=8):
    # Diagrama del MLP dibujado en código: círculos = neuronas, líneas = pesos.
    # Las capas anchas se truncan a max_neurons círculos con "⋮" en el medio.
    fig, ax = plt.subplots(figsize=(12, 6.5))
    xs = np.linspace(0.07, 0.93, len(layer_sizes))
    pos = []
    for n, x in zip(layer_sizes, xs):
        shown = min(n, max_neurons)
        ys = np.linspace(0.86, 0.16, shown)
        pos.append((x, ys, n > max_neurons))
    for (x0, ys0, _), (x1, ys1, _) in zip(pos[:-1], pos[1:]):
        for y0 in ys0:
            for y1 in ys1:
                ax.plot([x0, x1], [y0, y1], color="#c8c8c8", lw=0.35, zorder=1)
    for li, ((x, ys, trunc), n) in enumerate(zip(pos, layer_sizes)):
        color = ("#1f77b4" if li == 0 else
                 "#d62728" if li == len(layer_sizes) - 1 else "#2ca02c")
        mid = len(ys) // 2
        for k, y in enumerate(ys):
            if trunc and k == mid:
                ax.text(x, y, "⋮", ha="center", va="center", fontsize=18, zorder=3)
            else:
                ax.scatter([x], [y], s=420, color=color, edgecolor="white",
                           linewidth=1.5, zorder=2)
        ax.text(x, 0.93, f"{n} unidades" if n > 1 else "1 unidad",
                ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, 0.055, layer_labels[li], ha="center", va="top", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(-0.14, 1); ax.axis("off")
    fig.tight_layout()
    return fig

n_feat = Xc.shape[1]
arch_labels = [f"Entrada\n{n_feat} features\n(estandarizadas)",
               f"Oculta 1\nLinear({n_feat}→64)\nBatchNorm + ReLU\nDropout {HP_CLF['dropout']}",
               "Oculta 2\nLinear(64→32)\nBatchNorm + ReLU\nDropout " + str(HP_CLF['dropout']),
               "Salida\nLinear(32→1)\n1 logit → sigmoid\n(BCEWithLogitsLoss)"]
fig_arch = draw_mlp_architecture([n_feat, 64, 32, 1], arch_labels,
                                 "Arquitectura del MLP final — clasificación breast cancer")

with mlflow.start_run(run_id=run.info.run_id):
    mlflow.log_figure(fig_arch, "plots/arquitectura_mlp.png")
print("Diagrama de arquitectura enviado a MLflow: plots/arquitectura_mlp.png")
plt.show()
"""),
    ("md", r"""## 9. Resumen

- Una **neurona** calcula $z=\mathbf{w}^\top\mathbf{x}+b$ y $a=\phi(z)$; apilando
  capas con no linealidades obtenemos un **MLP** capaz de aproximar funciones
  complejas.
- Las **activaciones** y sus derivadas explican el **desvanecimiento del
  gradiente**; **ReLU**, BatchNorm, conexiones residuales e inicialización cuidada
  lo mitigan.
- **Decide la arquitectura antes de programar**: profundidad, ancho, activación, y
  sobre todo el par **salida ↔ pérdida** (lineal+MSE para regresión,
  sigmoid/softmax+BCE/cross-entropy para clasificación).
- **Backprop** = regla de la cadena capa por capa; el autograd lo hace por ti.
- **SGD** baja la superficie de pérdida; la **learning rate** es el knob más
  sensible; **momentum** y **Adam** mejoran la convergencia.
- Combina **dropout + weight decay + early stopping + batchnorm** contra el
  sobreajuste.
- Registra cada run en **MLflow** y guarda el ganador. Inspecciona en
  **http://localhost:5000**.
"""),
]

if __name__ == "__main__":
    build("01_regularization.ipynb", nb01)
    build("02_ensembles.ipynb", nb02)
    build("03_svm.ipynb", nb03)
    build("04_unsupervised.ipynb", nb04)
    build("05_neural_networks_pytorch.ipynb", nb05)
    print("\nTodos los notebooks generados.")
