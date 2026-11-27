"""
Trae solo la calificacion promedio y el numero de resenas (proxy de ventas) de cada SKU del catalogo
desde el widget de reviews.io de alkosto.com. Una consulta por producto.
Uso: python scripts/stats_resenas.py
"""
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
API = "https://api.reviews.io/timeline/data"


def stats(sku):
    q = dict(type="product_review", store="www.alkosto.com", page=1, per_page=1, sku=sku, lang="es")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "Mozilla/5.0"})
    for i in range(3):
        try:
            s = json.load(urllib.request.urlopen(req, timeout=30)).get("stats", {})
            return dict(sku=sku, calificacion=s.get("average_rating"), num_resenas=s.get("review_count"))
        except Exception:
            time.sleep(2 * (i + 1))
    return dict(sku=sku, calificacion=None, num_resenas=None)


cat = pd.read_csv(BASE / "data_reales" / "catalogo_alkosto.csv", dtype={"sku": str})
with ThreadPoolExecutor(4) as ex:
    filas = list(ex.map(stats, cat["sku"]))
df = pd.DataFrame(filas)
df.to_csv(BASE / "data_reales" / "resenas_stats.csv", index=False)
print(df.shape, "con resenas:", (df["num_resenas"].fillna(0) > 0).sum())
