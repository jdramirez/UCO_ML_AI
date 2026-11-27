"""
Normaliza (224x224, sin duplicados) solo las imagenes crudas que aun no estan en etiquetas.csv,
las agrega con etiqueta vacia y crea hojas de contacto nuevas (hoja_n<k>.jpg) para etiquetarlas.

Uso: python scripts/agregar_nuevas_imagenes.py [prefijo_hojas]
"""
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import preparar_imagenes_reales as P  # noqa: E402

prefijo = sys.argv[1] if len(sys.argv) > 1 else "hoja_n"
rows = list(csv.DictReader(open(P.ETIQUETAS)))
ya = {r["archivo"] for r in rows}
hashes = {P._ahash(Image.open(P.NORM / f)) for f in ya if (P.NORM / f).exists()}
nuevos, dup = [], 0
for f in sorted(Path(P.REALES / "crudas").glob("*.jpg")):
    if f.stem + ".jpg" in ya:
        continue
    try:
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    except Exception:
        continue
    if min(im.size) < 120:
        continue
    h = P._ahash(im)
    if h in hashes:
        dup += 1
        continue
    hashes.add(h)
    ImageOps.fit(im, (224, 224), Image.LANCZOS).save(P.NORM / (f.stem + ".jpg"), quality=90)
    nuevos.append(f.stem + ".jpg")
inicio = max([int(r["indice"]) for r in rows] + [-1]) + 1
for i, a in enumerate(nuevos):
    rows.append({"indice": inicio + i, "archivo": a, "tipo_imagen": ""})
w = csv.DictWriter(open(P.ETIQUETAS, "w", newline=""), fieldnames=["indice", "archivo", "tipo_imagen"])
w.writeheader()
w.writerows(rows)
cols, lado, por = 8, 160, 48
for k in range(0, len(nuevos), por):
    g = nuevos[k:k + por]
    hoja = Image.new("RGB", (cols * lado, ((len(g) + cols - 1) // cols) * lado), "white")
    d = ImageDraw.Draw(hoja)
    for i, a in enumerate(g):
        x, y = (i % cols) * lado, (i // cols) * lado
        hoja.paste(Image.open(P.NORM / a).resize((lado - 4, lado - 4)), (x + 2, y + 2))
        d.rectangle([x + 2, y + 2, x + 40, y + 20], fill="black")
        d.text((x + 5, y + 5), str(inicio + k + i), fill="yellow")
    hoja.save(P.HOJAS / f"{prefijo}{k // por:02d}.jpg", quality=85)
print(f"nuevas {len(nuevos)} (indices {inicio}-{inicio + len(nuevos) - 1}), duplicadas {dup}, "
      f"hojas {(len(nuevos) + por - 1) // por}")
