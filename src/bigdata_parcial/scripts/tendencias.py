"""Resume las tendencias de la base: likes relativos a la cuenta por tipo de imagen y de mensaje.
Uso: python scripts/tendencias.py"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
d = pd.concat([pd.read_csv(BASE / "data/ads/ads_publicados.csv"),
               pd.read_csv(BASE / "data/ads/ads_nuevos.csv").merge(
                   pd.read_csv(BASE / "_solo_profesor/ads_nuevos_etiquetas.csv"), on="post_id")])
d["likes_rel"] = d["likes"] / d.groupby("cuenta")["likes"].transform("median")
print(f"posts: {len(d)}  cuentas: {d.cuenta.nunique()}")
print("\nPor tipo de imagen (likes relativos a la mediana de la cuenta):")
print(d.groupby("tipo_imagen").agg(n=("post_id", "size"), likes_rel_mediana=("likes_rel", "median"),
                                   likes_rel_media=("likes_rel", "mean"), pct_sobre_promedio=("poner_ad", "mean"))
      .round(2).to_string())
print("\nImagen x mensaje (mediana likes relativos, n):")
t = d.groupby(["tipo_imagen", "tipo_mensaje"]).agg(med=("likes_rel", "median"), n=("post_id", "size")).round(2)
print(t.to_string())
print("\nPor cuenta e imagen (mediana likes relativos):")
print(d.pivot_table(index="cuenta", columns="tipo_imagen", values="likes_rel", aggfunc="median").round(2).to_string())
from scipy.stats import kruskal
grupos = [g["likes_rel"].values for _, g in d.groupby("tipo_imagen")]
print("\nKruskal-Wallis likes_rel ~ tipo_imagen: p =", round(kruskal(*grupos).pvalue, 4))
