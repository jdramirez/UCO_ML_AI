"""
Junta los datos de entrenamiento de los clasificadores:

  Mensajes (LSTM / regresion logistica):
     textos de anuncios de la Meta Ad Library (data_reales/textos_anuncios_reales.csv)
   + textos de publicaciones de Instagram etiquetadas (data_reales/ig_etiquetas.csv)
   -> data/textos/mensajes_train.csv          (fuente, id, cuenta, texto, tipo_mensaje)
   -> _solo_profesor/mensajes_test_completo.csv  test oculto (lo usa _solo_profesor/preparar_test_oculto.py)
   El test oculto sale SOLO de la Meta Ad Library: los textos de Instagram tambien estan en
   data/ads/*.csv con su tipo_mensaje, asi que su etiqueta se podria buscar. Se agrupa por plantilla
   (primeros 50 caracteres) para que un mismo texto repetido no quede en train y test a la vez.

  Imagenes (CNN):
     imagenes de la Ad Library etiquetadas (data_reales/etiquetas.csv, ya en data_reales/normalizadas/)
   + portadas de Instagram etiquetadas (data_reales/ig_portadas/) -> se normalizan a 224x224 como IG_<post_id>.jpg
   y luego: python scripts/preparar_imagenes_reales.py construir   (split agrupado train/test)

Uso: python scripts/construir_datos_modelos.py
"""
import csv
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parents[1]
R = BASE / "data_reales"


def main():
    # ---------------- mensajes
    meta = pd.read_csv(R / "textos_anuncios_reales.csv", dtype={"library_id": str})
    meta = meta.assign(fuente="meta_ad_library", id=meta["library_id"])[["fuente", "id", "cuenta", "texto",
                                                                         "tipo_mensaje"]]
    et = pd.read_csv(R / "ig_etiquetas.csv")
    cuentas = pd.read_csv(R / "instagram_cuentas.csv").set_index("cuenta_ig")["marca"]
    posts = {p["url"].rstrip("/").split("/")[-1]: p for p in json.load(open(R / "instagram_posts.json"))}
    filas = []
    for r in et[et["tipo_imagen"] != "descartar"].itertuples():  # solo posts validos (electro, Colombia)
        p = posts.get(r.post_id)
        m = re.search(r': "(.*)"', (p or {}).get("d", ""), re.S)
        if m and m.group(1).strip():
            filas.append(dict(fuente="instagram", id=r.post_id, cuenta=cuentas.get(r.cuenta_ig, r.cuenta_ig),
                              texto=m.group(1).strip(), tipo_mensaje=r.tipo_mensaje))
    msgs = pd.concat([meta, pd.DataFrame(filas)], ignore_index=True)
    msgs["_k"] = msgs["texto"].str.lower().str.replace(r"\W", "", regex=True).str[:120]
    msgs = msgs.drop_duplicates("_k").drop(columns="_k")
    dividir_mensajes(msgs.reset_index(drop=True))

    # ---------------- imagenes: agrega portadas de Instagram al conjunto etiquetado
    filas_img = list(csv.DictReader(open(R / "etiquetas.csv")))
    ya = {f["archivo"] for f in filas_img}
    sig = max(int(f["indice"]) for f in filas_img) + 1
    for r in et.itertuples():
        if r.tipo_imagen == "descartar":
            continue
        src = R / "ig_portadas" / f"{r.post_id}.jpg"
        dst_name = f"IG_{r.post_id}.jpg"
        if not src.exists() or dst_name in ya:
            continue
        ImageOps.fit(Image.open(src).convert("RGB"), (224, 224), Image.LANCZOS).save(R / "normalizadas" / dst_name,
                                                                                    quality=90)
        filas_img.append({"indice": sig, "archivo": dst_name, "tipo_imagen": r.tipo_imagen})
        sig += 1
    w = csv.DictWriter(open(R / "etiquetas.csv", "w", newline=""), fieldnames=["indice", "archivo", "tipo_imagen"])
    w.writeheader()
    w.writerows(filas_img)
    print("imagenes etiquetadas:", pd.Series([f["tipo_imagen"] for f in filas_img]).value_counts().to_dict())


def dividir_mensajes(msgs, seed=42):
    from sklearn.model_selection import StratifiedGroupKFold
    meta = msgs[msgs["fuente"] == "meta_ad_library"]
    plantilla = meta["texto"].str.lower().str.replace(r"\W", "", regex=True).str[:50]
    _, i_te = next(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
                   .split(meta, meta["tipo_mensaje"], groups=plantilla))
    idx_te = meta.index[i_te]
    test = msgs.loc[idx_te].sample(frac=1, random_state=seed).reset_index(drop=True)
    test.insert(0, "id_texto", [f"txt_{i:04d}" for i in range(len(test))])
    train = msgs.drop(idx_te).reset_index(drop=True)
    d = BASE / "data" / "textos"
    d.mkdir(parents=True, exist_ok=True)
    for viejo in ("mensajes_etiquetados.csv", "mensajes_test.csv"):
        (d / viejo).unlink(missing_ok=True)
    train.to_csv(d / "mensajes_train.csv", index=False)
    test[["id_texto", "cuenta", "texto", "tipo_mensaje", "id"]].rename(columns={"id": "library_id"}).to_csv(
        BASE / "_solo_profesor" / "mensajes_test_completo.csv", index=False)
    print("mensajes train:", train.shape, train["tipo_mensaje"].value_counts().to_dict(),
          train["fuente"].value_counts().to_dict())
    print("mensajes test :", test.shape, test["tipo_mensaje"].value_counts().to_dict())


if __name__ == "__main__":
    main()
