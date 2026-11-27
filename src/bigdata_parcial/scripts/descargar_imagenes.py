"""
Descarga las imagenes listadas en un JSON (recolectado de la Biblioteca de Anuncios de Meta)
a una carpeta de imagenes crudas.

Uso:  python scripts/descargar_imagenes.py data_reales/anuncios_meta.json data_reales/crudas
JSON: lista de {"id": ..., "src": url, ...}
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def bajar(item, destino):
    f = destino / f"{item['id']}.jpg"
    if f.exists():
        return True
    try:
        req = urllib.request.Request(item["src"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            f.write_bytes(r.read())
        return True
    except Exception:
        return False


def main(json_path, destino):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    items = json.load(open(json_path))
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(lambda it: bajar(it, destino), items))
    print(f"descargadas {sum(ok)} de {len(items)} en {destino}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
