"""
Variables del modelo de decision. Las MISMAS funciones se usan para construir el dataset
(scripts/construir_dataset_ads.py) y en el simulador (app.py), para que el modelo vea lo mismo
en entrenamiento y en produccion.

Solo se usan variables que se conocen ANTES de publicar: imagen, texto, producto y momento de
publicacion. Los likes y comentarios son el RESULTADO (objetivo), nunca variables.
"""
import re

import pandas as pd

CATEGORIAS_PRODUCTO = {
    "nevera": ["nevera", "nevecon", "nevecón", "refrigerador", "refrigeracion", "refrigeración", "congelador"],
    "lavadora": ["lavadora", "lavasecadora", "secadora", "lavado"],
    "televisor": ["televisor", "tv", "smart tv", "pantalla", "qled", "oled", "mini led", "google tv"],
    "aire_acondicionado": ["aire acondicionado", "aires acondicionados", "minisplit", "btu"],
    "freidora_aire": ["freidora", "airfryer", "air fryer"],
    "licuadora": ["licuadora", "batidora", "procesador"],
    "microondas": ["microondas"],
    "estufa": ["estufa", "cubierta", "horno", "campana"],
    "ventilador": ["ventilador"],
    "pequenos_electro": ["cafetera", "olla", "arrocera", "waflera", "sanduchera", "plancha", "aspiradora",
                         "hervidor", "tostadora", "sartén", "sarten"],
    "calentador": ["calentador", "agua caliente"],
    "computador": ["portátil", "portatil", "computador", "laptop", "macbook", "notebook"],
    "celular": ["celular", "celulares", "smartphone", "iphone", "galaxy", "teléfono", "telefono"],
    "audio_tecnologia": ["parlante", "audífonos", "audifonos", "monitor", "tablet", "proyector",
                         "barra de sonido", "smartwatch", "reloj"],
}
TEMPORADAS = ["normal", "black_friday", "navidad", "prima", "dia_madre", "dia_padre", "amor_amistad"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def _primera_mencion(texto):
    t = str(texto).lower()
    for cat, palabras in CATEGORIAS_PRODUCTO.items():
        for p in palabras:
            if re.search(r"(^|\W)" + re.escape(p) + r"(\W|$)", t):
                return cat, p
    return "marca_institucional", "ninguno"


def categoria_desde_texto(texto):
    return _primera_mencion(texto)[0]


def producto_desde_texto(texto):
    """palabra del producto mencionado en el texto (ej. 'freidora', 'nevecon', 'cafetera')."""
    return _primera_mencion(texto)[1].replace("ó", "o").replace("é", "e").replace("í", "i")


def _n_dia_semana(anio, mes, dia_semana, n):
    """n-esimo dia de la semana del mes (dia_semana: lunes=0 ... domingo=6)."""
    d = pd.Timestamp(anio, mes, 1)
    return d + pd.Timedelta(days=(dia_semana - d.dayofweek) % 7 + 7 * (n - 1))


def temporada_de(fecha):
    """Temporada comercial en Colombia segun la fecha de publicacion. Las fechas especiales cuentan desde
    dos semanas antes (cuando las marcas empiezan a pautar) hasta el mismo dia."""
    d = pd.Timestamp(fecha).normalize()
    a = d.year

    def antes(dia, semanas=2):
        return dia - pd.Timedelta(weeks=semanas) <= d <= dia

    if antes(_n_dia_semana(a, 5, 6, 2)):           # dia de la madre: 2do domingo de mayo
        return "dia_madre"
    if antes(_n_dia_semana(a, 6, 6, 3)):           # dia del padre: 3er domingo de junio
        return "dia_padre"
    if antes(_n_dia_semana(a, 9, 5, 3)):           # amor y amistad: 3er sabado de septiembre
        return "amor_amistad"
    if (d.month == 11 and d.day >= 15) or (d.month == 12 and d.day <= 2):  # Black Friday / Black Days
        return "black_friday"
    if (d.month == 6 and d.day >= 20) or (d.month == 12 and 15 <= d.day <= 20):  # pago de la prima
        return "prima"
    if d.month == 12:
        return "navidad"
    return "normal"


def variables_texto(texto):
    t = str(texto)
    precios = re.findall(r"\$\s?\d[\d.,]{3,}", t)
    pct = [int(x) for x in re.findall(r"(\d{1,2})\s?%", t)]
    return {
        "num_palabras": len(t.split()),
        "num_hashtags": t.count("#"),
        "num_emojis": len(EMOJI.findall(t)),
        "menciona_precio": int(bool(precios)),
        "descuento_mencionado": max(pct) if pct else 0,
        "tiene_pregunta": int("?" in t),
        "es_concurso": int(bool(re.search(r"concurso|sorteo|particip|gana(dor|r)?\b|regalamos|giveaway|"
                                          r"comenta y|etiqueta a", t, re.I))),
        "llamado_accion": int(bool(re.search(r"c[oó]mpra|encu[eé]ntra|con[oó]ce|visita|aprovecha|ll[eé]va|"
                                             r"estrena|desliza|link|bio", t, re.I))),
    }


def construir_fila(texto, tipo_imagen, tipo_mensaje, cuenta, seguidores, fecha, es_carrusel, n_imagenes,
                   categoria=None, precio_referencia=None, calificacion_producto=None, num_resenas=None,
                   tasa_historica_categoria=None, producto=None, es_video=0):
    """tasa_historica_categoria: tasa de interaccion promedio de los posts ANTERIORES de la misma cuenta y
    categoria (proxy del 'promedio de clicks por producto'). En el simulador se toma del historico."""
    fecha = pd.Timestamp(fecha)
    fila = {
        "cuenta": cuenta, "seguidores": float(seguidores),
        "tipo_imagen": tipo_imagen, "tipo_mensaje": tipo_mensaje,
        "categoria_producto": categoria or categoria_desde_texto(texto),
        "menciona_producto": int((categoria or categoria_desde_texto(texto)) != "marca_institucional"),
        "producto_mencionado": producto or producto_desde_texto(texto),
        "tasa_historica_categoria": tasa_historica_categoria,
        "temporada": temporada_de(fecha), "dia_semana": int(fecha.dayofweek), "hora_publicacion": int(fecha.hour),
        "es_carrusel": int(es_carrusel), "n_imagenes": int(n_imagenes), "es_video": int(es_video),
        "precio_referencia": precio_referencia, "calificacion_producto": calificacion_producto,
        "num_resenas": num_resenas,
    }
    fila.update(variables_texto(texto))
    return fila


CATEGORICAS = ["cuenta", "tipo_imagen", "tipo_mensaje", "categoria_producto", "producto_mencionado", "temporada"]
NUMERICAS = ["seguidores", "menciona_producto", "tasa_historica_categoria", "dia_semana", "hora_publicacion", "es_carrusel", "n_imagenes", "es_video", "precio_referencia",
             "calificacion_producto", "num_palabras", "num_hashtags", "num_emojis",
             "menciona_precio", "descuento_mencionado", "tiene_pregunta", "es_concurso", "llamado_accion"]
# num_resenas queda en la base solo para exploracion (H5): NO es variable del modelo.
FEATURES = CATEGORICAS + NUMERICAS
