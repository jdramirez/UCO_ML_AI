"""
Combina lo recolectado de la Biblioteca de Anuncios de Meta (Colombia) en:
  data_reales/imagenes_meta.json  -> una entrada por imagen (id = AD<library_id>_<n>)
  data_reales/textos_anuncios.csv -> una fila por anuncio con su texto (copy)

Entradas:
  data_reales/anuncios_meta.json      -> anuncios por cuenta (Kalley, Haceb, Challenger, Mabe, LG, Imusa, Alkosto)
  data_reales/anuncios_meta_raw.json  -> anuncios de busquedas por palabra clave (mucho ruido: se filtran a mano)
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1] / "data_reales"


def limpiar(t):
    t = re.sub(r"[​]", "", t or "")
    partes = re.split(r"\n(?:Sponsored|Publicidad)\n", t, maxsplit=1)
    cuerpo = partes[-1] if len(partes) > 1 else t
    return re.sub(r"\n{2,}", "\n", cuerpo).strip()


def meta(t):
    t = t or ""
    g = lambda p: (re.search(p, t) or [None, None])[1]
    return dict(estado="activo" if re.search(r"\nActive\n", t) else "inactivo",
                fecha_inicio=g(r"Started running on ([A-Za-z]{3} \d{1,2}, \d{4})"),
                library_id=g(r"Library ID:\s*(\d+)"),
                anunciante=g(r"\n([^\n]+)\n(?:Sponsored|Publicidad)\n"))


def main():
    ads = json.load(open(BASE / "anuncios_meta.json"))
    raw = json.load(open(BASE / "anuncios_meta_raw.json")) if (BASE / "anuncios_meta_raw.json").exists() else []
    items, vistos, filas = [], set(), []
    for a in ads:
        m = meta(a["txt"])
        filas.append(dict(library_id=a["library_id"], cuenta=a["cuenta"], fuente="cuenta", estado=m["estado"],
                          fecha_inicio=m["fecha_inicio"], n_imagenes=len(a["imgs"]), n_videos=a["n_videos"],
                          texto=limpiar(a["txt"])))
        for k, im in enumerate(a["imgs"]):
            key = im["src"].split("?")[0]
            if key in vistos:
                continue
            vistos.add(key)
            items.append(dict(id=f"AD{a['library_id']}_{k}", src=im["src"], fuente="cuenta", cuenta=a["cuenta"],
                              library_id=a["library_id"], w=im["w"], h=im["h"]))
    for r in raw:
        key = r["src"].split("?")[0]
        m = meta(r["txt"])
        if key in vistos or not m["library_id"]:
            continue
        vistos.add(key)
        items.append(dict(id=f"AD{m['library_id']}_k", src=r["src"], fuente="busqueda", cuenta=m["anunciante"],
                          library_id=m["library_id"], query=r["query"], w=r["w"], h=r["h"]))
        filas.append(dict(library_id=m["library_id"], cuenta=m["anunciante"], fuente="busqueda", estado=m["estado"],
                          fecha_inicio=m["fecha_inicio"], n_imagenes=1, n_videos=0, texto=limpiar(r["txt"]),
                          query=r["query"]))
    c, n = Counter(i["id"] for i in items), Counter()
    for i in items:
        if c[i["id"]] > 1:
            n[i["id"]] += 1
            i["id"] = f"{i['id']}{n[i['id']]}"
    json.dump(items, open(BASE / "imagenes_meta.json", "w"), ensure_ascii=False, indent=0)
    df = pd.DataFrame(filas).drop_duplicates("library_id")
    df.to_csv(BASE / "textos_anuncios.csv", index=False)
    print(len(items), "imagenes;", len(df), "anuncios;", df[df.fuente == "cuenta"].cuenta.value_counts().to_dict())


if __name__ == "__main__":
    main()
