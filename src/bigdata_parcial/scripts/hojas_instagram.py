"""
Descarga la imagen de portada de cada post de Instagram y arma hojas de contacto numeradas + una tabla
de textos para etiquetar a mano tipo_imagen y tipo_mensaje (data_reales/ig_etiquetas.csv).
Uso: python scripts/hojas_instagram.py <cuenta_ig>
"""
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

BASE = Path(__file__).resolve().parents[1]
R = BASE / "data_reales"
cuenta = sys.argv[1]
posts = [p for p in json.load(open(R / "instagram_posts.json")) if p.get("cuenta_ig") == cuenta and p.get("d")]
posts.sort(key=lambda p: p.get("t", ""))
cdir = R / "ig_portadas"
cdir.mkdir(exist_ok=True)


def bajar(p):
    pid = p["url"].rstrip("/").split("/")[-1]
    f = cdir / f"{pid}.jpg"
    src = p.get("img")  # og:image = portada real del post
    if not f.exists() and src:
        try:
            f.write_bytes(urllib.request.urlopen(urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"}),
                                                 timeout=30).read())
        except Exception:
            return None
    return pid if f.exists() else None


with ThreadPoolExecutor(8) as ex:
    ids = list(ex.map(bajar, posts))
cols, lado, por = 8, 160, 48
validos = [(i, pid) for i, pid in enumerate(ids) if pid]
for k in range(0, len(validos), por):
    g = validos[k:k + por]
    hoja = Image.new("RGB", (cols * lado, ((len(g) + cols - 1) // cols) * lado), "white")
    d = ImageDraw.Draw(hoja)
    for j, (i, pid) in enumerate(g):
        x, y = (j % cols) * lado, (j // cols) * lado
        hoja.paste(ImageOps.fit(Image.open(cdir / f"{pid}.jpg").convert("RGB"), (lado - 4, lado - 4)), (x + 2, y + 2))
        d.rectangle([x + 2, y + 2, x + 40, y + 20], fill="black")
        d.text((x + 5, y + 5), str(i), fill="yellow")
    hoja.save(R / "hojas" / f"ig_{cuenta}_{k // por:02d}.jpg", quality=85)
with open(R / f"ig_textos_{cuenta}.txt", "w") as fh:
    for i, p in enumerate(posts):
        m = re.search(r': "(.*)"', p["d"], re.S)
        fh.write(f"{i} | {ids[i]} | {' '.join((m.group(1) if m else '').split())[:220]}\n")
print(len(posts), "posts;", (len(validos) + por - 1) // por, "hojas")
