"""
Clasificador del TEXTO del anuncio (copy): ¿hacia donde va el mensaje?

    promocion        -> marca, tienda, evento, financiacion, envio, obsequios, invitacion a comprar
    caracteristicas  -> especificaciones y beneficios del producto (litros, pulgadas, inverter, potencia...)
    descuento        -> precio, % de descuento, antes/ahora

Interfaz:
    predict_proba(textos) -> DataFrame con columnas p_promocion, p_caracteristicas, p_descuento
    explicar(texto)       -> DataFrame [termino, peso] para la clase predicha
    pesos_globales(n)     -> DataFrame [clase, termino, peso]
    nombre -> str

- ClasificadorMensajeReglas : MOCK con listas de palabras clave.
- ClasificadorMensajeLR     : Pipeline sklearn (vectorizer + LogisticRegression) del estudiante.
"""
import numpy as np
import pandas as pd

from . import rutas
from .texto import normalizar, softmax as _softmax, tokens

CLASES = rutas.CLASES_MENSAJE
COLS = [f"p_{c}" for c in CLASES]


class ClasificadorMensajeReglas:
    nombre = "Reglas (mock: palabras clave)"
    PALABRAS = {
        "promocion": {"compra": 0.8, "compralo": 0.8, "tienda": 1.0, "tiendas": 1.0, "visita": 1.0, "envio": 0.8,
                      "gratis": 0.6, "cuotas": 1.0, "financiacion": 1.0, "credito": 1.0, "obsequio": 1.2,
                      "regalo": 1.2, "conoce": 0.8, "encuentra": 0.8, "aprovecha": 0.5, "black": 0.4,
                      "days": 0.4, "hogar": 0.4, "registrate": 1.0, "beneficios": 0.8},
        "caracteristicas": {"litros": 1.5, "pulgadas": 1.5, "inverter": 1.5, "kilos": 1.2, "kg": 1.2,
                            "potencia": 1.2, "capacidad": 1.2, "tecnologia": 1.0, "resolucion": 1.2,
                            "4k": 1.2, "energia": 1.0, "velocidades": 1.2, "acero": 0.8, "puestos": 1.0,
                            "silenciosa": 0.8, "diseno": 0.6, "funciones": 1.0, "sonido": 0.8, "w": 0.6},
        "descuento": {"descuento": 1.5, "descuentos": 1.5, "dto": 1.5, "off": 1.2, "antes": 1.2, "ahora": 0.8,
                      "precio": 1.0, "precios": 1.0, "rebaja": 1.5, "rebajas": 1.5, "oferta": 1.0,
                      "ofertas": 1.0, "ahorra": 0.8, "solo": 0.4, "desde": 0.6},
    }

    def _scores(self, texto):
        tk = tokens(texto)
        z = [sum(self.PALABRAS[c].get(t, 0) for t in tk) for c in CLASES]
        z[CLASES.index("descuento")] += 1.0 * str(texto).count("%")
        return z

    def predict_proba(self, textos):
        return pd.DataFrame(_softmax([self._scores(t) for t in textos]), columns=COLS)

    def explicar(self, texto):
        c = CLASES[int(np.argmax(self._scores(texto)))]
        fil = [(t, self.PALABRAS[c][t]) for t in tokens(texto) if t in self.PALABRAS[c]]
        if c == "descuento" and "%" in str(texto):
            fil.append(("%", 1.0 * str(texto).count("%")))
        return pd.DataFrame(fil, columns=["termino", "peso"]).drop_duplicates()

    def pesos_globales(self, n=10):
        fil = [(c, t, w) for c in CLASES for t, w in sorted(self.PALABRAS[c].items(), key=lambda x: -x[1])[:n]]
        return pd.DataFrame(fil, columns=["clase", "termino", "peso"])


class ClasificadorMensajeLR:
    """Pipeline([("vectorizer", TfidfVectorizer|CountVectorizer), ("clf", LogisticRegression)]) guardado en
    modelos/entrenados/mensaje_lr.joblib. Etiquetas: 'promocion', 'caracteristicas', 'descuento'."""

    def __init__(self, ruta=rutas.MENSAJE_LR):
        import joblib
        self.pipe = joblib.load(ruta)
        self.vec, self.clf = self.pipe.steps[0][1], self.pipe.steps[-1][1]
        self.clases = [str(c) for c in self.clf.classes_]
        try:
            self.vocab = np.array(self.vec.get_feature_names_out())
        except AttributeError:
            self.vocab = np.array(self.vec.get_feature_names())
        self.nombre = f"Regresion logistica ({type(self.vec).__name__}, {len(self.vocab)} terminos)"

    def predict_proba(self, textos):
        p = self.pipe.predict_proba(list(textos))
        df = pd.DataFrame(0.0, index=range(len(p)), columns=COLS)
        for j, c in enumerate(self.clases):
            if f"p_{c}" in df:
                df[f"p_{c}"] = p[:, j]
        return df

    def explicar(self, texto):
        k = int(np.argmax(self.pipe.predict_proba([texto])[0]))
        x = self.vec.transform([texto]).tocoo()
        df = pd.DataFrame({"termino": self.vocab[x.col], "peso": x.data * self.clf.coef_[k][x.col]})
        return df.reindex(df["peso"].abs().sort_values(ascending=False).index)

    def pesos_globales(self, n=10):
        fil = []
        for k, c in enumerate(self.clases):
            for i in np.argsort(self.clf.coef_[k])[-n:][::-1]:
                fil.append((c, self.vocab[i], self.clf.coef_[k][i]))
        return pd.DataFrame(fil, columns=["clase", "termino", "peso"])


class ClasificadorMensajeLSTM:
    """Modelo keras que recibe TEXTO CRUDO (TextVectorization adentro) y devuelve softmax de 3 clases en el
    orden de mensaje_lstm_config.json. No es explicable por palabra: el tablero muestra solo las probabilidades."""

    def __init__(self, ruta):
        import json

        import tensorflow as tf
        self.tf = tf
        self.model = tf.keras.models.load_model(ruta)
        cfg = json.loads(rutas.MENSAJE_LSTM_CONFIG.read_text()) if rutas.MENSAJE_LSTM_CONFIG.exists() else {}
        self.clases = cfg.get("clases", CLASES)
        self.normalizar = cfg.get("normalizar_texto", False)
        self.nombre = "LSTM del estudiante"

    def predict_proba(self, textos):
        textos = [normalizar(t) if self.normalizar else str(t) for t in textos]
        p = self.model.predict(self.tf.constant(textos), verbose=0)
        df = pd.DataFrame(0.0, index=range(len(p)), columns=COLS)
        for j, c in enumerate(self.clases):
            df[f"p_{c}"] = p[:, j]
        return df

    def explicar(self, texto):
        return pd.DataFrame(columns=["termino", "peso"])

    def pesos_globales(self, n=10):
        return pd.DataFrame(columns=["clase", "termino", "peso"])


def clasificadores_disponibles():
    """nombre -> fabrica. El tablero deja escoger entre la LSTM (modelo principal) y la regresion logistica
    (linea base explicable)."""
    disp = {"Reglas (mock)": lambda: ClasificadorMensajeReglas()}
    if rutas.MENSAJE_LR.exists():
        disp["Regresion logistica (estudiante)"] = lambda: ClasificadorMensajeLR()
    lstm = rutas.primero_que_exista(rutas.MENSAJE_LSTM)
    if lstm is not None:
        disp["LSTM (estudiante)"] = lambda: ClasificadorMensajeLSTM(lstm)
    return disp


def cargar_clasificador_mensaje():
    disp = clasificadores_disponibles()
    return list(disp.values())[-1]()


__all__ = ["cargar_clasificador_mensaje", "normalizar"]
