"""
Construye la BASE REAL de publicaciones de marcas de electrodomesticos en Instagram.

Entradas
  data_reales/instagram_posts.json      -> posts recolectados (texto, likes, comentarios, fecha, imagenes)
  data_reales/instagram_cuentas.csv     -> cuenta_ig, marca, seguidores
  data_reales/ig_etiquetas.csv          -> post_id, tipo_imagen, tipo_mensaje (etiquetado a mano)
  data_reales/catalogo_alkosto.csv + resenas_stats.csv -> precio, calificacion y resenas por marca/categoria

Salidas
  data/ads/ads_publicados.csv   -> posts con resultado (likes, comentarios, tasa_interaccion, poner_ad)
  data/ads/ads_nuevos.csv       -> 20% de los posts SIN resultado (para predecir)
  _solo_profesor/ads_nuevos_etiquetas.csv
  data/ads/imagenes_posts/<post_id>_<k>.jpg   (imagenes de cada post; carrusel = varias)

Metricas (ver ENUNCIADO, seccion de metricas):
  tasa_interaccion = (likes + comentarios) / seguidores      -> proxy del CTR
  likes = ventas_proxy                                       -> proxy de VENTAS (real)
  tasa_conversion_proxy = likes / impresiones_estimadas      -> proxy de la tasa de conversion
  alcance_estimado, impresiones_estimadas                    -> ESTIMADOS con benchmarks (Instagram no los publica)
  poner_ad = 1 si tasa_interaccion > promedio de la tasa de su cuenta
  tasa_historica_categoria = tasa media de los posts ANTERIORES de la misma cuenta y categoria

Uso: python scripts/construir_dataset_ads.py
"""
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from modelos.variables import construir_fila  # noqa: E402

REALES = BASE / "data_reales"


def parsear_descripcion(d):
    """'58 likes, 6 comments - kalleyhogar on July 8, 2025: "texto". '"""
    m = re.match(r'\s*([\d,.KkMm]+) likes?, ([\d,.KkMm]+) comments? - (\S+) on ([^:]+): "(.*)"\.?\s*$', d or "", re.S)
    if not m:
        return None

    def num(x):
        x = x.replace(",", "")
        mult = 1000 if x[-1] in "Kk" else 1_000_000 if x[-1] in "Mm" else 1
        return int(float(x.rstrip("KkMm")) * mult)
    return dict(likes=num(m.group(1)), comentarios=num(m.group(2)), texto=m.group(5).strip())


def referencia_producto(catalogo, marca, categoria):
    # 1) misma marca y categoria  2) misma categoria (otras marcas)  3) sin producto -> nulo
    c = catalogo[(catalogo["marca"].str.lower() == str(marca).lower()) & (catalogo["categoria"] == categoria)]
    if len(c) < 2:
        c = catalogo[catalogo["categoria"] == categoria]
    if c.empty:
        return None, None, None
    return float(c["precio"].median()), float(c["calificacion"].median()), float(c["num_opiniones"].median())


def bajar(url, destino):
    if destino.exists():
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        destino.write_bytes(urllib.request.urlopen(req, timeout=30).read())
        return True
    except Exception:
        return False


def main():
    posts = json.load(open(REALES / "instagram_posts.json"))
    cuentas = pd.read_csv(REALES / "instagram_cuentas.csv").set_index("cuenta_ig")
    etiquetas = pd.read_csv(REALES / "ig_etiquetas.csv").set_index("post_id")
    catalogo = pd.read_csv(REALES / "catalogo_alkosto.csv", dtype={"sku": str})
    st = pd.read_csv(REALES / "resenas_stats.csv", dtype={"sku": str})
    catalogo = catalogo.merge(st, on="sku", how="left")
    catalogo["calificacion"] = catalogo["calificacion_y"].fillna(catalogo["calificacion_x"])
    catalogo["num_opiniones"] = catalogo["num_resenas"].fillna(catalogo["num_opiniones"])

    from collections import Counter
    frecuencia = Counter(i["src"].split("?")[0] for p in posts for i in p.get("imgs", []))
    img_dir = BASE / "data" / "ads" / "imagenes_posts"
    img_dir.mkdir(parents=True, exist_ok=True)
    filas, descargas = [], []
    for p in posts:
        info = parsear_descripcion(p.get("d"))
        post_id = p["url"].rstrip("/").split("/")[-1]
        if not info or post_id not in etiquetas.index or not p.get("t"):
            continue
        et = etiquetas.loc[post_id]
        if et["tipo_imagen"] == "descartar":
            continue
        cta = cuentas.loc[p["cuenta_ig"]]
        # portada (og:image) + imagenes del carrusel, quitando imagenes que se repiten en muchos posts
        # (elementos fijos de la pagina de Instagram, no del post)
        extra = [i["src"] for i in p.get("imgs", []) if frecuencia[i["src"].split("?")[0]] <= 2]
        imgs = [p.get("img")] + [u for u in extra if u.split("?")[0] != (p.get("img") or "").split("?")[0]]
        fecha = pd.Timestamp(p["t"]).tz_convert("America/Bogota").tz_localize(None)
        es_video = int(p.get("es_video", 0) or "/reel/" in p["url"])
        fila = construir_fila(info["texto"], et["tipo_imagen"], et["tipo_mensaje"], cta["marca"], cta["seguidores"],
                              fecha, len(imgs) > 1 and not es_video, 1 if es_video else len(imgs), es_video=es_video)
        precio, calif, resenas = referencia_producto(catalogo, cta["marca"], fila["categoria_producto"])
        fila.update(precio_referencia=precio, calificacion_producto=calif, num_resenas=resenas)
        archivos = []
        for k, u in enumerate(imgs):
            f = img_dir / f"{post_id}_{k}.jpg"
            descargas.append((u, f))
            archivos.append(f.name)
        filas.append(dict(post_id=post_id, url=p["url"], fecha=fecha, texto=info["texto"],
                          imagenes=";".join(archivos), **fila,
                          likes=info["likes"], comentarios=info["comentarios"]))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda x: bajar(*x), descargas))

    df = pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)
    df["interacciones"] = df["likes"] + df["comentarios"]
    df["tasa_interaccion"] = df["interacciones"] / df["seguidores"]
    df["tasa_promedio_cuenta"] = df.groupby("cuenta")["tasa_interaccion"].transform("mean")
    df["poner_ad"] = (df["tasa_interaccion"] > df["tasa_promedio_cuenta"]).astype(int)

    # --- "promedio de clicks por producto": tasa media de los posts ANTERIORES de la misma cuenta y categoria
    #     (expanding mean desplazado una posicion -> no usa el propio post ni posts futuros)
    df["tasa_historica_categoria"] = df.groupby(["cuenta", "categoria_producto"])["tasa_interaccion"].transform(
        lambda s: s.expanding().mean().shift(1))
    previa_cuenta = df.groupby("cuenta")["tasa_interaccion"].transform(lambda s: s.expanding().mean().shift(1))
    df["tasa_historica_categoria"] = df["tasa_historica_categoria"].fillna(previa_cuenta)  # 1er post de la categoria

    # --- impresiones y alcance ESTIMADOS (Instagram no los publica). Benchmarks:
    #     imagen estatica ~7% de los seguidores en cuentas pequenas, decrece con el tamano de la cuenta;
    #     los carruseles alcanzan mas (Socialinsider 2026); un post con mas interaccion que lo normal se
    #     distribuye mas; impresiones = alcance x frecuencia (1.15-1.40).
    rng_imp = np.random.default_rng(7)
    tasa_alcance = 0.07 * (df["seguidores"] / 3000) ** -0.15
    viral = (df["tasa_interaccion"] / df["tasa_promedio_cuenta"]).clip(0.25, 9) ** 0.5
    df["alcance_estimado"] = (df["seguidores"] * tasa_alcance * np.where(df["es_carrusel"] == 1, 1.25, 1.0) * viral *
                              rng_imp.lognormal(0, 0.15, len(df))).round()
    df["alcance_estimado"] = np.maximum(df["alcance_estimado"], df["interacciones"] * 1.5).round()
    df["impresiones_estimadas"] = (df["alcance_estimado"] * rng_imp.uniform(1.15, 1.40, len(df))).round()
    df["tasa_interaccion_alcance"] = df["interacciones"] / df["alcance_estimado"]
    # likes = proxy de VENTAS; tasa de conversion proxy = likes / impresiones
    df["ventas_proxy"] = df["likes"]
    df["tasa_conversion_proxy"] = df["likes"] / df["impresiones_estimadas"]

    rng = np.random.default_rng(2026)
    nuevo = rng.random(len(df)) < 0.2
    resultado = ["likes", "comentarios", "interacciones", "tasa_interaccion", "tasa_promedio_cuenta", "poner_ad",
                 "alcance_estimado", "impresiones_estimadas", "tasa_interaccion_alcance", "ventas_proxy",
                 "tasa_conversion_proxy"]
    df[~nuevo].to_csv(BASE / "data" / "ads" / "ads_publicados.csv", index=False)
    df[nuevo].drop(columns=resultado).to_csv(BASE / "data" / "ads" / "ads_nuevos.csv", index=False)
    (BASE / "_solo_profesor").mkdir(exist_ok=True)
    df[nuevo][["post_id"] + resultado].to_csv(BASE / "_solo_profesor" / "ads_nuevos_etiquetas.csv", index=False)
    validar(df, nuevo)


def validar(df, nuevo):
    """Chequeos de sentido comun sobre la base."""
    print(f"posts: {len(df)}  publicados: {(~nuevo).sum()}  nuevos: {nuevo.sum()}")
    print(df.groupby("cuenta").agg(posts=("post_id", "size"), seguidores=("seguidores", "first"),
                                   likes_mediana=("likes", "median"), tasa_media=("tasa_interaccion", "mean"),
                                   alcance_pct=("alcance_estimado", lambda a: (a / df.loc[a.index, "seguidores"]).median()),
                                   er_alcance=("tasa_interaccion_alcance", "median"),
                                   poner=("poner_ad", "mean")).round(4).to_string())
    chequeos = {
        "likes >= 0 y comentarios >= 0": bool((df["likes"] >= 0).all() and (df["comentarios"] >= 0).all()),
        "interacciones <= alcance <= impresiones": bool((df["interacciones"] <= df["alcance_estimado"]).all() and
                                                        (df["alcance_estimado"] <= df["impresiones_estimadas"]).all()),
        "alcance mediano entre 1% y 15% de seguidores": bool(
            0.01 <= (df["alcance_estimado"] / df["seguidores"]).median() <= 0.15),
        "tasa de interaccion (seguidores) mediana < 2%": bool(df["tasa_interaccion"].median() < 0.02),
        "fechas en orden y sin futuro": bool(df["fecha"].max() <= pd.Timestamp.now()),
        "sin post_id duplicados": bool(df["post_id"].is_unique),
    }
    for k, v in chequeos.items():
        print(("OK   " if v else "FALLA"), k)
    rho = df[["tasa_historica_categoria", "tasa_interaccion"]].corr("spearman").iloc[0, 1]
    print(f"INFO  Spearman(tasa_historica_categoria, tasa_interaccion) = {rho:.3f}  "
          f"(cercano a 0 = el historial de la categoria anticipa poco)")
    print(f"INFO  likes: mediana {df['likes'].median():.0f}, p95 {df['likes'].quantile(.95):.0f}, max {df['likes'].max()}"
          f"  -> distribucion muy sesgada (use log1p para predecir likes)")
    print(df.groupby(["tipo_imagen", "tipo_mensaje"])["poner_ad"].agg(["mean", "size"]).round(2).to_string())


if __name__ == "__main__":
    main()
