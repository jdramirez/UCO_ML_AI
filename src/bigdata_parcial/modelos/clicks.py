"""
Modelo de decision: ¿publicamos (pautamos) este ad?

Objetivo: poner_ad = 1 si la tasa de interaccion esperada supera el promedio de la cuenta.
  tasa_interaccion = (likes + comentarios) / seguidores   (proxy del CTR, ver ENUNCIADO)

Interfaz:
    predict_proba(df) -> np.array con P(poner_ad = 1)
    predict_likes(df)       -> np.array con los likes esperados, o None
    predict_impresiones(df) -> np.array con las impresiones esperadas, o None
    importancia()     -> DataFrame [variable, importancia]
    nombre -> str

El DataFrame de entrada tiene las columnas de modelos.variables.FEATURES (las arma construir_fila).

- ModeloAdsReglas  : MOCK con reglas de mercadeo "a ojo".
- ModeloAdsSklearn : Pipeline sklearn del estudiante (preprocesamiento + RandomForest).
"""
import numpy as np
import pandas as pd

from . import rutas
from .variables import CATEGORICAS, FEATURES, NUMERICAS  # noqa: F401  (se re-exportan)


class ModeloAdsReglas:
    """Reglas de mercadeo (mock). Supone que los descuentos y los concursos siempre funcionan y que las
    personas en la imagen ayudan un poco. No usa interacciones: el Random Forest debe superarlo."""

    nombre = "Reglas de mercadeo (mock)"
    PESOS = {"es_concurso": 0.30, "tipo_imagen": 0.20, "tipo_mensaje": 0.20, "descuento_mencionado": 0.15,
             "es_carrusel": 0.10, "hora_publicacion": 0.05}

    def _score(self, r):
        s = -0.2
        s += 1.5 * float(r["es_concurso"])
        s += {"personas": 0.3, "descuento": 0.2, "producto": 0.0}.get(r["tipo_imagen"], 0)
        s += {"descuento": 0.3, "promocion": 0.1, "caracteristicas": 0.0}.get(r["tipo_mensaje"], 0)
        s += 0.01 * float(r["descuento_mencionado"] or 0)
        s += 0.2 * float(r["es_carrusel"])
        s += 0.1 if 18 <= int(r["hora_publicacion"]) <= 21 else 0
        return s

    def predict_proba(self, df):
        s = df.apply(self._score, axis=1).values
        return 1 / (1 + np.exp(-2 * s))

    def _tasa(self, df):
        # tasa base tipica de una marca en Instagram (~0.05% de los seguidores) escalada por el puntaje
        return 0.0005 * np.exp(df.apply(self._score, axis=1).values)

    def predict_likes(self, df):
        return 0.95 * self._tasa(df) * df["seguidores"].values

    def predict_impresiones(self, df):
        seg = df["seguidores"].values
        alcance = seg * 0.07 * (seg / 3000) ** -0.15 * np.where(df["es_carrusel"].values == 1, 1.25, 1.0)
        return 1.25 * alcance

    def importancia(self):
        s = pd.Series(self.PESOS)
        return (s / s.sum()).sort_values(ascending=False).rename_axis("variable").reset_index(name="importancia")


class ModeloAdsSklearn:
    """Pipeline sklearn guardado en modelos/entrenados/ads_random_forest.joblib que recibe el DataFrame
    CRUDO con las columnas FEATURES (ColumnTransformer / OneHotEncoder / SimpleImputer adentro) y cuyo
    ultimo paso tiene feature_importances_. Regresores: ads_likes_regresor.joblib y ads_impresiones_regresor.joblib
    (si se entrenaron sobre log1p(y), guardelos dentro de un TransformedTargetRegressor)."""

    def __init__(self, ruta=rutas.ADS_CLF):
        import joblib
        self.pipe = joblib.load(ruta)
        self.reg_likes = joblib.load(rutas.ADS_REG_LIKES) if rutas.ADS_REG_LIKES.exists() else None
        self.reg_imp = joblib.load(rutas.ADS_REG_IMPRESIONES) if rutas.ADS_REG_IMPRESIONES.exists() else None
        self.nombre = f"{type(self.pipe.steps[-1][1]).__name__} del estudiante"
        if self.reg_likes is not None or self.reg_imp is not None:
            self.nombre += " + regresores"

    def predict_proba(self, df):
        return self.pipe.predict_proba(df[FEATURES])[:, 1]

    # Los regresores son opcionales (no se piden en el parcial): sin ellos, likes e impresiones se estiman por reglas.
    def predict_likes(self, df):
        if self.reg_likes is None:
            return ModeloAdsReglas().predict_likes(df)
        return np.clip(self.reg_likes.predict(df[FEATURES]), 0, None)

    def predict_impresiones(self, df):
        if self.reg_imp is None:
            return ModeloAdsReglas().predict_impresiones(df)
        return np.clip(self.reg_imp.predict(df[FEATURES]), 0, None)

    def _nombres(self):
        pre = self.pipe[:-1] if len(self.pipe.steps) > 1 else None
        if pre is None:
            return list(FEATURES)
        try:
            return list(pre.get_feature_names_out())
        except Exception:
            return None

    def importancia(self, agrupar=True):
        imp = getattr(self.pipe.steps[-1][1], "feature_importances_", None)
        if imp is None:
            return pd.DataFrame(columns=["variable", "importancia"])
        nombres = self._nombres()
        if nombres is None or len(nombres) != len(imp):
            nombres = [f"x{i}" for i in range(len(imp))]
        df = pd.DataFrame({"variable": nombres, "importancia": imp})
        if agrupar:  # suma las dummies de una misma variable (tipo_imagen_personas -> tipo_imagen)
            def original(n):
                n = n.split("__", 1)[-1]
                cand = [f for f in FEATURES if n == f or n.startswith(f + "_")]
                return max(cand, key=len) if cand else n
            df["variable"] = df["variable"].map(original)
            df = df.groupby("variable", as_index=False)["importancia"].sum()
        return df.sort_values("importancia", ascending=False).reset_index(drop=True)


def cargar_modelo_ads():
    if rutas.ADS_CLF.exists():
        return ModeloAdsSklearn()
    return ModeloAdsReglas()
