"""
Simulador de Ads - Electrodomesticos Colombia
Parcial Big Data / Machine Learning - UCO

Ejecutar:   streamlit run app.py

Un solo flujo: 1) imagen -> CNN (tipo de imagen)  2) texto -> LSTM / regresion logistica (tipo de mensaje)
3) producto y publicacion  4) Random Forest -> ¿la tasa de interaccion superara el promedio? -> ¿publicar?

Funciona desde el primer momento con modelos de REGLAS (mock). Cuando el estudiante guarda sus modelos
en modelos/entrenados/ con los nombres de ENUNCIADO.md, el simulador los carga automaticamente.
"""
import html
from datetime import date, time

import pandas as pd
import plotly.express as px
import streamlit as st

from modelos import rutas
from modelos.clicks import cargar_modelo_ads
from modelos.imagenes import abrir_imagen, cargar_clasificador
from modelos.mensaje import clasificadores_disponibles
from modelos.texto import normalizar
from modelos.variables import (CATEGORIAS_PRODUCTO, FEATURES, categoria_desde_texto, construir_fila,
                               producto_desde_texto)

st.set_page_config(page_title="Simulador de Ads", page_icon="📊", layout="wide")
AZUL = "#1f3b73"


# ------------------------------------------------------------------ carga (cache)
@st.cache_resource(show_spinner="Cargando clasificador de imagenes...")
def get_clasificador():
    return cargar_clasificador()


@st.cache_resource(show_spinner="Cargando clasificador de mensajes...")
def get_mensaje(nombre):
    return clasificadores_disponibles()[nombre]()


@st.cache_resource(show_spinner="Cargando modelo de decision...")
def get_modelo_ads():
    return cargar_modelo_ads()


@st.cache_data
def datos_referencia():
    pub = rutas.BASE / "data" / "ads" / "ads_publicados.csv"
    return pd.read_csv(pub) if pub.exists() else None


def etiqueta(nombre):
    return f"{'🟡' if 'mock' in nombre.lower() else '🟢'} {nombre}"


def resaltar(texto, pesos):
    partes = []
    for palabra in str(texto).split():
        w = pesos.get(normalizar(palabra).strip(".,!?¿¡:;()\"'*"), 0)
        if w == 0:
            partes.append(html.escape(palabra))
        else:
            alpha = min(0.85, 0.25 + abs(w) / 3)
            color = "31,59,115" if w > 0 else "214,69,69"
            partes.append(f"<span style='background:rgba({color},{alpha});color:white;padding:1px 4px;"
                          f"border-radius:4px' title='peso {w:+.2f}'>{html.escape(palabra)}</span>")
    return " ".join(partes)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 Simulador de Ads")
st.sidebar.caption("Electrodomesticos · Colombia")
opciones_msg = list(clasificadores_disponibles())
nombre_msg = st.sidebar.selectbox("Clasificador de mensajes", opciones_msg, index=len(opciones_msg) - 1,
                                  help="La LSTM es el modelo principal; la regresion logistica es la linea base "
                                       "explicable (muestra las palabras importantes).")
clasif_img = get_clasificador()
clasif_msg = get_mensaje(nombre_msg)
modelo_ads = get_modelo_ads()
st.sidebar.markdown("**Modelos activos**")
st.sidebar.markdown(f"- Imagen: {etiqueta(clasif_img.nombre)}\n- Mensaje: {etiqueta(clasif_msg.nombre)}\n"
                    f"- Decision: {etiqueta(modelo_ads.nombre)}")
st.sidebar.caption("🟡 = reglas (mock) · 🟢 = modelo entrenado por el estudiante. Guarde sus modelos en "
                   "`modelos/entrenados/` y recargue.")
if st.sidebar.button("🔄 Recargar modelos"):
    st.cache_resource.clear()
    st.rerun()

ref = datos_referencia()
st.title("¿Publicamos este ad?")
st.caption("Solo se usa lo que se conoce ANTES de publicar: la imagen, el texto, el producto y el momento de "
           "publicacion. Los likes y comentarios son el resultado que el modelo intenta anticipar.")

c_img, c_msg = st.columns(2)

# ------------------------------------------------------------------ 1. imagen
with c_img:
    st.subheader("1. Imagen")
    img_file = st.file_uploader("Suba la imagen del ad", type=["jpg", "jpeg", "png", "webp"])
    ejemplos = sorted((rutas.EJEMPLOS / "imagenes").glob("*.jpg"))
    ej = st.selectbox("...o use una imagen de ejemplo", ["(ninguna)"] + [e.name for e in ejemplos])
    img = abrir_imagen(img_file) if img_file is not None else (
        abrir_imagen(rutas.EJEMPLOS / "imagenes" / ej) if ej != "(ninguna)" else None)
    tipo_img_pred = None
    if img is not None:
        a, b = st.columns([1, 1])
        a.image(img, width="stretch")
        probs = clasif_img.predict_proba(img)
        tipo_img_pred = probs.idxmax()
        fig = px.bar(x=probs.values, y=list(probs.index), orientation="h", range_x=[0, 1],
                     labels={"x": "probabilidad", "y": ""}, title=f"Tipo de imagen: {tipo_img_pred}")
        fig.update_traces(marker_color=AZUL)
        fig.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10))
        b.plotly_chart(fig)
    tipos_img = rutas.CLASES_IMAGEN
    tipo_imagen = st.selectbox("Tipo de imagen usado en la prediccion", tipos_img,
                               index=tipos_img.index(tipo_img_pred) if tipo_img_pred in tipos_img else 0)
    n_imagenes = st.number_input("Numero de imagenes (carrusel)", 1, 10, 1)
    es_video = st.checkbox("Es un video / reel (la imagen es la portada)", False)

# ------------------------------------------------------------------ 2. mensaje
with c_msg:
    st.subheader("2. Mensaje")
    ejemplos_txt = []
    if ref is not None:
        ejemplos_txt = ref.drop_duplicates("texto").sample(min(25, len(ref)), random_state=3)["texto"].tolist()
    ej_txt = st.selectbox("Texto real de ejemplo", ["(escribir)"] + ejemplos_txt,
                          format_func=lambda t: t if t == "(escribir)" else t[:70].replace("\n", " ") + "...")
    texto = st.text_area("Texto del ad", "" if ej_txt == "(escribir)" else ej_txt, height=150,
                         placeholder="¡Black Days Kalley! Lleva tu freidora de aire de 6L con 56% de descuento...")
    tipo_msg_pred = None
    if texto.strip():
        pm = clasif_msg.predict_proba([texto]).iloc[0]
        pm.index = [i.replace("p_", "") for i in pm.index]
        tipo_msg_pred = pm.idxmax()
        expl = clasif_msg.explicar(texto)
        if len(expl):
            st.markdown(f"<div style='line-height:1.9'>{resaltar(texto, dict(zip(expl['termino'], expl['peso'])))}"
                        f"</div>", unsafe_allow_html=True)
        fig = px.bar(x=pm.values, y=list(pm.index), orientation="h", range_x=[0, 1],
                     labels={"x": "probabilidad", "y": ""}, title=f"Tipo de mensaje: {tipo_msg_pred}")
        fig.update_traces(marker_color=AZUL)
        fig.update_layout(height=220, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig)
    tipos_msg = rutas.CLASES_MENSAJE
    tipo_mensaje = st.selectbox("Tipo de mensaje usado en la prediccion", tipos_msg,
                                index=tipos_msg.index(tipo_msg_pred) if tipo_msg_pred in tipos_msg else 0)

# ------------------------------------------------------------------ 3. producto y publicacion
st.subheader("3. Producto y publicacion")
cuentas = (ref.groupby("cuenta")["seguidores"].max().to_dict() if ref is not None else {"Kalley": 246_000})
categorias = list(CATEGORIAS_PRODUCTO) + ["marca_institucional"]
cat_auto = categoria_desde_texto(texto) if texto.strip() else "marca_institucional"
a, b, c, d = st.columns(4)
cuenta = a.selectbox("Marca / cuenta", list(cuentas))
seguidores = b.number_input("Seguidores de la cuenta", 1_000, 50_000_000, int(cuentas[cuenta]), step=1_000)
categoria = c.selectbox("Categoria del producto", categorias, index=categorias.index(cat_auto),
                        help="Se detecta del texto; puede cambiarla.")
ref_cat = ref[(ref["cuenta"] == cuenta) & (ref["categoria_producto"] == categoria)] if ref is not None else None
precio_def = float(ref_cat["precio_referencia"].median()) if ref_cat is not None and len(ref_cat) and \
    ref_cat["precio_referencia"].notna().any() else 1_000_000.0
precio = d.number_input("Precio de referencia (COP)", 0, 30_000_000, int(precio_def), step=50_000)
a, c, d = st.columns(3)
calif = a.slider("Calificacion del producto (Alkosto)", 1.0, 5.0, 4.8, 0.1)
fecha = c.date_input("Fecha de publicacion", date.today())
hora = d.slider("Hora de publicacion", 0, 23, 19)
umbral = st.slider("Umbral de decision P(tasa > promedio)", 0.1, 0.9, 0.5, 0.05)

hist_cat = None
if ref is not None:
    h = ref[(ref["cuenta"] == cuenta) & (ref["categoria_producto"] == categoria)]["tasa_interaccion"]
    hist_cat = float(h.mean()) if len(h) else float(ref[ref["cuenta"] == cuenta]["tasa_interaccion"].mean()) \
        if (ref["cuenta"] == cuenta).any() else None
producto = producto_desde_texto(texto) if texto.strip() else "ninguno"
st.caption(f"Producto detectado en el texto: **{producto}** · tasa historica de la categoria en la cuenta: "
           f"**{hist_cat:.3%}**" if hist_cat else f"Producto detectado: **{producto}**")
fila = pd.DataFrame([construir_fila(texto, tipo_imagen, tipo_mensaje, cuenta, seguidores,
                                    pd.Timestamp.combine(fecha, time(hour=hora)), n_imagenes > 1, n_imagenes,
                                    categoria, precio, calif, None, hist_cat, producto, es_video)])[FEATURES]
with st.expander("Variables que recibe el modelo"):
    st.dataframe(fila.T.astype(str).rename(columns={0: "valor"}), width="stretch")

# ------------------------------------------------------------------ 4. resultado
st.divider()
st.subheader("4. Resultado")
p = float(modelo_ads.predict_proba(fila)[0])
likes = modelo_ads.predict_likes(fila)
imp = modelo_ads.predict_impresiones(fila)
ref_c = ref[ref["cuenta"] == cuenta] if ref is not None else None
m1, m2, m3, m4 = st.columns(4)
m1.metric("P(tasa de interaccion > promedio)", f"{p:.0%}")
m2.metric("Ventas esperadas (proxy: likes)", f"{likes[0]:,.0f}" if likes is not None else "—",
          help=(f"Mediana de la cuenta: {ref_c['likes'].median():,.0f}. " if ref_c is not None and len(ref_c) else "")
          + "Estimacion de referencia por reglas.")
m3.metric("Impresiones esperadas", f"{imp[0]:,.0f}" if imp is not None else "—",
          help="Estimadas: Instagram no publica las impresiones (ver ENUNCIADO)")
m4.metric("Conversion proxy (likes / impresiones)", f"{likes[0] / imp[0]:.2%}" if likes is not None and
          imp is not None and imp[0] > 0 else "—", help="Proxy de la tasa de conversion")
if p >= umbral:
    st.success(f"✅ **Publicar / pautar.** P={p:.0%} ≥ umbral {umbral:.0%}")
else:
    st.error(f"⛔ **No publicar todavia.** P={p:.0%} < umbral {umbral:.0%}. Pruebe otra imagen o mensaje.")

s1, s2 = st.columns(2)
sim = pd.concat([fila.assign(tipo_imagen=ti, tipo_mensaje=tm) for ti in tipos_img for tm in tipos_msg],
                ignore_index=True)
sim["p"] = modelo_ads.predict_proba(sim)
mat = sim.pivot(index="tipo_imagen", columns="tipo_mensaje", values="p").reindex(index=tipos_img, columns=tipos_msg)
fig = px.imshow(mat, text_auto=".0%", zmin=0, zmax=1, color_continuous_scale="Blues", aspect="auto",
                labels={"x": "tipo de mensaje", "y": "tipo de imagen", "color": "P"},
                title="Simulacion: combinaciones de imagen y mensaje")
fig.update_layout(height=360)
s1.plotly_chart(fig)
horas = pd.concat([fila.assign(hora_publicacion=h) for h in range(24)], ignore_index=True)
horas["p"] = modelo_ads.predict_proba(horas)
fig = px.line(horas, x="hora_publicacion", y="p", markers=True, range_y=[0, 1],
              title="Sensibilidad: ¿a que hora publicar?", labels={"p": "P", "hora_publicacion": "hora"})
fig.add_hline(y=umbral, line_dash="dash")
fig.update_traces(line_color=AZUL)
fig.update_layout(height=360)
s2.plotly_chart(fig)

imp = modelo_ads.importancia()
if len(imp):
    fig = px.bar(imp.head(15).sort_values("importancia"), x="importancia", y="variable", orientation="h",
                 title=f"Importancia de variables — {modelo_ads.nombre}")
    fig.update_traces(marker_color=AZUL)
    fig.update_layout(height=430)
    st.plotly_chart(fig)
