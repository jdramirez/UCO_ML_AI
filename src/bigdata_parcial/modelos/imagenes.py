"""
Clasificacion del tipo de imagen del ad: personas | producto | descuento.

Interfaz:
    predict_proba(imagen_PIL) -> pd.Series indexada por clase (suma 1)
    nombre -> str

- ClasificadorImagenReglas : MOCK basado en estadisticas de color (lo que deben reemplazar).
- ClasificadorImagenCNN    : carga la CNN keras del estudiante + cnn_config.json
"""
import json

import numpy as np
import pandas as pd
from PIL import Image

from . import rutas


class ClasificadorImagenReglas:
    """Reglas de color hechas a mano (mock):
       - colores muy saturados (naranja, rojo, amarillo) o fondo negro -> descuento
       - mucho fondo blanco/gris claro                                  -> producto
       - mucho tono de piel                                             -> personas
    Es intencionalmente simple: una CNN entrenada debe superarlo."""

    nombre = "Reglas de color (mock)"

    def caracteristicas(self, img):
        im = img.convert("RGB").resize((64, 64))
        hsv = np.asarray(im.convert("HSV"), dtype=float) / 255.0
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        return {
            "claro": np.mean((v > 0.8) & (s < 0.15)),
            "saturado": np.mean((s > 0.7) & (v > 0.5)),
            "oscuro": np.mean(v < 0.15),
            "piel": np.mean((h < 0.1) & (s > 0.2) & (s < 0.6) & (v > 0.35)),
        }

    def predict_proba(self, img):
        f = self.caracteristicas(img)
        z = np.array([
            5.0 * f["piel"] - 0.8,                                 # personas
            3.0 * f["claro"] - 2.0 * f["piel"],                    # producto
            3.0 * f["saturado"] + 1.5 * f["oscuro"] + 0.3,         # descuento
        ])
        p = np.exp(z - z.max())
        return pd.Series(p / p.sum(), index=["personas", "producto", "descuento"]).reindex(rutas.CLASES_IMAGEN)


class ClasificadorImagenCNN:
    """Carga modelos/entrenados/cnn_imagenes.keras (o .h5) y cnn_config.json:
        {"img_size": [64, 64], "color": "rgb" | "gris", "clases": ["personas","producto","descuento"], "normalizar": true}
    'normalizar' = true si el estudiante dividio los pixeles por 255 FUERA del modelo.
    Si el modelo ya tiene una capa Rescaling adentro use "normalizar": false."""

    def __init__(self, ruta):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(ruta)
        cfg = json.loads(rutas.CNN_CONFIG.read_text()) if rutas.CNN_CONFIG.exists() else {}
        self.size = tuple(cfg.get("img_size", [64, 64]))
        self.color = cfg.get("color", "rgb")
        self.clases = cfg.get("clases", rutas.CLASES_IMAGEN)
        self.normalizar = cfg.get("normalizar", True)
        self.nombre = f"CNN del estudiante ({ruta.name}, {self.size[0]}x{self.size[1]} {self.color})"

    def predict_proba(self, img):
        im = img.convert("L" if self.color == "gris" else "RGB").resize(self.size)
        x = np.asarray(im, dtype="float32")
        if self.color == "gris":
            x = x[..., None]
        if self.normalizar:
            x = x / 255.0
        p = self.model.predict(x[None, ...], verbose=0)[0]
        if p.min() < 0 or abs(p.sum() - 1) > 1e-3:  # salida en logits
            p = np.exp(p - p.max())
            p = p / p.sum()
        return pd.Series(p, index=self.clases)


def cargar_clasificador():
    ruta = rutas.primero_que_exista(rutas.CNN_MODELO)
    if ruta is not None:
        return ClasificadorImagenCNN(ruta)
    return ClasificadorImagenReglas()


def abrir_imagen(archivo):
    return Image.open(archivo).convert("RGB")
