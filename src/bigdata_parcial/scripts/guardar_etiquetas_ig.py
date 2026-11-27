"""
Guarda etiquetas manuales de una cuenta de Instagram en data_reales/ig_etiquetas.csv.
Uso: python scripts/guardar_etiquetas_ig.py <cuenta_ig> "<0h 1p 2d 3x ...>" "<0p 1c 2d ...>"
  imagen: h=personas p=producto d=descuento x=descartar   mensaje: p=promocion c=caracteristicas d=descuento
Los indices son los de data_reales/ig_textos_<cuenta>.txt (generado por hojas_instagram.py).
"""
import collections
import csv
import os
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "data_reales"
cuenta, img, msg = sys.argv[1], sys.argv[2], sys.argv[3]
ci = {"h": "personas", "p": "producto", "d": "descuento", "x": "descartar"}
cm = {"p": "promocion", "c": "caracteristicas", "d": "descuento"}
I = {int(t[:-1]): ci[t[-1]] for t in img.split()}
M = {int(t[:-1]): cm[t[-1]] for t in msg.split()}
filas = []
for linea in open(R / f"ig_textos_{cuenta}.txt"):
    i, pid, _ = linea.split(" | ", 2)
    i = int(i)
    if pid == "None" or i not in I:
        continue
    filas.append({"post_id": pid, "cuenta_ig": cuenta, "tipo_imagen": I[i], "tipo_mensaje": M[i]})
f = R / "ig_etiquetas.csv"
otras = [r for r in csv.DictReader(open(f)) if r["cuenta_ig"] != cuenta] if os.path.exists(f) else []
w = csv.DictWriter(open(f, "w", newline=""), fieldnames=["post_id", "cuenta_ig", "tipo_imagen", "tipo_mensaje"])
w.writeheader()
w.writerows(otras + filas)
print(cuenta, len(filas), dict(collections.Counter(r["tipo_imagen"] for r in filas)),
      dict(collections.Counter(r["tipo_mensaje"] for r in filas if r["tipo_imagen"] != "descartar")))
