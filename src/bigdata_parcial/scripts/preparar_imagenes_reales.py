"""
Prepara el dataset de imagenes REALES de ads (descargadas de Google Imagenes, Instagram, etc.).

Pasos:
  1) normalizar : toma las imagenes crudas, elimina duplicados y las deja del mismo tamano
                  (recorte centrado a cuadrado + resize) en data_reales/normalizadas/
  2) hojas      : crea hojas de contacto numeradas para etiquetar a ojo
                  y una plantilla etiquetas.csv (archivo, tipo_imagen) para llenar
  3) construir  : con etiquetas.csv lleno, arma data/imagenes/train/<clase>/ y data/imagenes/test/
                  (split agrupado) y guarda las etiquetas del test en _solo_profesor/; despues
                  _solo_profesor/preparar_test_oculto.py deja el test como oculto

Uso:
  python scripts/preparar_imagenes_reales.py normalizar <carpeta_crudas> [--size 224] [--modo recorte|relleno]
  python scripts/preparar_imagenes_reales.py hojas
  python scripts/preparar_imagenes_reales.py construir [--test 0.2]

Clases validas en etiquetas.csv: personas, producto, descuento  (o "descartar")
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

BASE = Path(__file__).resolve().parents[1]
REALES = BASE / "data_reales"
NORM = REALES / "normalizadas"
HOJAS = REALES / "hojas"
ETIQUETAS = REALES / "etiquetas.csv"
CLASES = ["personas", "producto", "descuento"]
EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"}


def _ahash(im, n=8):
    """hash perceptual simple para detectar imagenes repetidas (aunque cambie el tamano)."""
    g = im.convert("L").resize((n, n), Image.BILINEAR)
    px = list(g.getdata())
    m = sum(px) / len(px)
    return "".join("1" if p > m else "0" for p in px)


def normalizar(crudas, size=224, modo="recorte", min_lado=120):
    NORM.mkdir(parents=True, exist_ok=True)
    vistos, n_ok, n_dup, n_mal = set(), 0, 0, 0
    for f in sorted(Path(crudas).rglob("*")):
        if f.suffix.lower() not in EXT:
            continue
        try:
            im = Image.open(f)
            im = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            n_mal += 1
            continue
        if min(im.size) < min_lado:
            n_mal += 1
            continue
        h = _ahash(im)
        if h in vistos:
            n_dup += 1
            continue
        vistos.add(h)
        if modo == "recorte":   # recorte centrado al cuadrado (pierde bordes de banners muy anchos)
            im = ImageOps.fit(im, (size, size), Image.LANCZOS)
        else:                   # relleno: conserva toda la imagen y completa con el color del borde
            im = ImageOps.pad(im, (size, size), Image.LANCZOS, color=im.getpixel((0, 0)))
        nombre = f.stem + ".jpg"  # conserva el ID del anuncio (ej. AD<library_id>_0.jpg)
        im.save(NORM / nombre, quality=90)
        n_ok += 1
    print(f"normalizadas: {n_ok}  duplicadas: {n_dup}  descartadas (corruptas o muy pequenas): {n_mal}")


def hojas(por_hoja=48, cols=8, lado=160):
    HOJAS.mkdir(parents=True, exist_ok=True)
    archivos = sorted(NORM.glob("*.jpg"))
    for k in range(0, len(archivos), por_hoja):
        grupo = archivos[k:k + por_hoja]
        filas = (len(grupo) + cols - 1) // cols
        hoja = Image.new("RGB", (cols * lado, filas * lado), "white")
        d = ImageDraw.Draw(hoja)
        for i, f in enumerate(grupo):
            x, y = (i % cols) * lado, (i // cols) * lado
            hoja.paste(Image.open(f).resize((lado - 4, lado - 4)), (x + 2, y + 2))
            d.rectangle([x + 2, y + 2, x + 40, y + 20], fill="black")
            d.text((x + 5, y + 5), str(k + i), fill="yellow")
        hoja.save(HOJAS / f"hoja_{k // por_hoja:02d}.jpg", quality=85)
    if not ETIQUETAS.exists():
        with open(ETIQUETAS, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["indice", "archivo", "tipo_imagen"])
            for i, f in enumerate(archivos):
                w.writerow([i, f.name, ""])
    print(f"{len(archivos)} imagenes -> {len(list(HOJAS.glob('*.jpg')))} hojas en {HOJAS}; llene {ETIQUETAS}")


def _dhash(im, n=16):
    g = im.convert("L").resize((n + 1, n), Image.BILINEAR)
    px = list(g.getdata())
    return sum(1 << k for k, (a, b) in enumerate((px[r * (n + 1) + c], px[r * (n + 1) + c + 1])
                                                for r in range(n) for c in range(n)) if a > b)


def _grupos(archivos, umbral=40):
    """agrupa imagenes casi iguales (mismo creativo con otro recorte/texto) para que no queden
    unas en train y otras en test (fuga de informacion)."""
    hs = [_dhash(Image.open(NORM / a)) for a in archivos]
    padre = list(range(len(archivos)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            if bin(hs[i] ^ hs[j]).count("1") <= umbral:
                padre[raiz(i)] = raiz(j)
    return [raiz(i) for i in range(len(archivos))]


def construir(test=0.2, seed=2026):
    filas = [r for r in csv.DictReader(open(ETIQUETAS)) if r["tipo_imagen"] in CLASES]
    rnd = random.Random(seed)
    train_dir, test_dir = BASE / "data" / "imagenes" / "train", BASE / "data" / "imagenes" / "test"
    for d in (train_dir, test_dir):
        if d.exists():
            shutil.rmtree(d)
    test_dir.mkdir(parents=True)
    etiquetas_test = []
    for c in CLASES:
        (train_dir / c).mkdir(parents=True)
        grupo = [r for r in filas if r["tipo_imagen"] == c]
        gid = _grupos([r["archivo"] for r in grupo])
        ids = sorted(set(gid))
        rnd.shuffle(ids)
        objetivo, en_test = round(len(grupo) * test), set()
        for g in ids:  # llenar el test con grupos completos hasta llegar a la proporcion
            if sum(gid.count(x) for x in en_test) >= objetivo:
                break
            en_test.add(g)
        n_tr = n_te = 0
        for r, g in zip(grupo, gid):
            if g in en_test:
                etiquetas_test.append((r["archivo"], c))
                n_te += 1
            else:
                shutil.copy(NORM / r["archivo"], train_dir / c / r["archivo"])
                n_tr += 1
        print(f"{c:10s} train={n_tr:4d}  test={n_te:4d}  (grupos de imagenes casi iguales: {len(grupo) - len(ids)})")
    rnd.shuffle(etiquetas_test)
    (BASE / "_solo_profesor").mkdir(exist_ok=True)
    with open(BASE / "_solo_profesor" / "imagenes_test_completo.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["archivo", "tipo_imagen", "archivo_original"])
        for k, (archivo, c) in enumerate(etiquetas_test):
            nuevo = f"img_{k:04d}.jpg"
            shutil.copy(NORM / archivo, test_dir / nuevo)
            w.writerow([nuevo, c, archivo])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["normalizar", "hojas", "construir"])
    ap.add_argument("crudas", nargs="?")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--modo", choices=["recorte", "relleno"], default="recorte")
    ap.add_argument("--test", type=float, default=0.2)
    a = ap.parse_args()
    if a.paso == "normalizar":
        normalizar(a.crudas, a.size, a.modo)
    elif a.paso == "hojas":
        hojas()
    else:
        construir(a.test)
