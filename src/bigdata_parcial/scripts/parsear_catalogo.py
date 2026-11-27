"""
Convierte las tarjetas del buscador de Alkosto (data_reales/catalogo_alkosto_raw.json) en una tabla:
sku, nombre, marca, categoria, precio_antes, precio, descuento_pct, calificacion, num_opiniones, url
Uso: python scripts/parsear_catalogo.py
"""
import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MARCAS = ["HACEB", "KALLEY", "LG", "SAMSUNG", "MABE", "CHALLENGER", "WHIRLPOOL", "ELECTROLUX", "OSTER", "IMUSA",
          "BLACK+DECKER", "BLACK & DECKER", "NINJA", "HISENSE", "TCL", "PANASONIC", "HYUNDAI", "SAMURAI", "UNIVERSAL",
          "KITCHENAID", "T-FAL", "POWERXL", "PHILIPS", "SONY", "CAIXUN", "JVC", "CUISINART", "HAMILTON BEACH",
          "ABBA", "INDURAMA", "CENTRALES", "INSIGNIA", "MIDEA", "BOSCH", "AIWA", "TEKA", "TOSHIBA", "XIAOMI",
          "ROKU", "PRACTIMAX", "INNOVA", "HOLSTEIN", "UMCO", "SANKEY", "HP", "LENOVO", "ASUS", "ACER", "APPLE",
          "DELL", "MSI", "JBL", "MOTOROLA", "BOSE", "HONOR", "REALME", "XIAOMI", "VIVO", "OPPO", "SHARK", "KRUPS"]


def precio(s):
    return float(s.replace("$", "").replace(".", "").replace(",", "").strip())


def parsear(c):
    t = c["t"]
    lineas = [l.strip() for l in t.split("\n") if l.strip()]
    nombre = next((l for l in lineas if len(l) > 25 and "$" not in l), lineas[0])
    up = t.upper()
    marca = next((m for m in MARCAS if re.search(r"(^|\W)" + re.escape(m) + r"(\W|$)", up)), None)
    if marca is None:
        cand = re.findall(r"\b[A-Z][A-Z0-9+&\-]{2,}\b", nombre)
        marca = cand[0] if cand else "OTRA"
    marca = {"BLACK & DECKER": "BLACK+DECKER"}.get(marca, marca)
    cal = re.search(r"\n(\d\.\d)\n\((\d[\d.,]*)\)", t)
    desc = re.search(r"-(\d{1,2})%", t) or re.search(r"(\d{1,2})\n%\n+DTO", t)
    precios = [precio(p) for p in re.findall(r"\$\s?\n?[\d.]{4,}", t.replace("$\n", "$"))]
    if not precios:
        return None
    antes = max(precios)
    d = int(desc.group(1)) if desc else 0
    return dict(sku=c["sku"], nombre=nombre, marca=marca.title() if marca not in ("LG", "TCL", "JVC", "HP", "JBL", "MSI") else marca,
                categoria=c["categoria"], precio_antes=antes, precio=round(antes * (1 - d / 100), -2),
                descuento_pct=d, calificacion=float(cal.group(1)) if cal else None,
                num_opiniones=int(cal.group(2).replace(".", "").replace(",", "")) if cal else 0,
                patrocinado=bool(c.get("ads")), url=c["href"])


def main():
    raw = json.load(open(BASE / "data_reales" / "catalogo_alkosto_raw.json"))
    df = pd.DataFrame([x for x in (parsear(c) for c in raw) if x])
    df.to_csv(BASE / "data_reales" / "catalogo_alkosto.csv", index=False)
    print(df.shape)
    print(df.groupby("categoria").agg(n=("sku", "size"), precio_med=("precio", "median"),
                                      opiniones_med=("num_opiniones", "median"), cal=("calificacion", "mean")))
    print(df.marca.value_counts().head(15).to_dict())


if __name__ == "__main__":
    main()
