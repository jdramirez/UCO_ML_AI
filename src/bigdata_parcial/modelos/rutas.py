"""Rutas y nombres de archivo que el tablero busca. NO cambiar los nombres:
el tablero carga automaticamente los modelos entrenados si existen en modelos/entrenados/."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ENTRENADOS = BASE / "modelos" / "entrenados"
EJEMPLOS = BASE / "ejemplos_para_subir"

# --- Modelo 1: CNN de tipo de imagen
CNN_MODELO = [ENTRENADOS / "cnn_imagenes.keras", ENTRENADOS / "cnn_imagenes.h5"]
CNN_CONFIG = ENTRENADOS / "cnn_config.json"

# --- Modelo 2: tipo de mensaje del texto del anuncio
#     2a LSTM (modelo keras que recibe texto crudo)  -> modelo principal
MENSAJE_LSTM = [ENTRENADOS / "mensaje_lstm.keras", ENTRENADOS / "mensaje_lstm"]
MENSAJE_LSTM_CONFIG = ENTRENADOS / "mensaje_lstm_config.json"
#     2b Regresion logistica (Pipeline sklearn: vectorizer + LogisticRegression) -> linea base explicable
MENSAJE_LR = ENTRENADOS / "mensaje_lr.joblib"

# --- Modelo 3: decision de poner el ad (Pipeline sklearn con preprocesamiento + RandomForest/arboles)
ADS_CLF = ENTRENADOS / "ads_random_forest.joblib"
# --- Regresores: likes esperados e impresiones esperadas (Pipelines sklearn con las mismas FEATURES)
ADS_REG_LIKES = ENTRENADOS / "ads_likes_regresor.joblib"
ADS_REG_IMPRESIONES = ENTRENADOS / "ads_impresiones_regresor.joblib"

CLASES_IMAGEN = ["personas", "producto", "descuento"]
CLASES_MENSAJE = ["promocion", "caracteristicas", "descuento"]


def primero_que_exista(rutas):
    for r in rutas:
        if Path(r).exists():
            return Path(r)
    return None
