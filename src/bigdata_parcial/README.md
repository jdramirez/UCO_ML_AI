# Simulador de Ads – Parcial Big Data / ML

Un solo flujo para decidir si publicar un ad de electrodomésticos en Colombia:

1. **Imagen** → una **CNN** dice si muestra `personas`, `producto` o `descuento`.
2. **Texto** → una **LSTM** (o la regresión logística, como línea base explicable) dice si el mensaje es de `promocion`, `caracteristicas` o `descuento`.
3. **Producto y publicación** → marca, seguidores, categoría, precio, calificación y reseñas en Alkosto, fecha y hora.
4. **Random Forest** → P(la tasa de interacción supera el promedio de la cuenta), más la simulación imagen × mensaje y por hora. Los likes e impresiones esperados se muestran como referencia con una estimación por reglas.

Al inicio funciona con **modelos de reglas (mock)**. Los estudiantes entrenan sus modelos y los guardan en `modelos/entrenados/`; el simulador los carga solo.

👉 **Las instrucciones del parcial están en [ENUNCIADO.md](ENUNCIADO.md).**

## Instalación

```bash
cd src/bigdata_parcial
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Datos (todos reales)

| Archivo | Contenido |
|---|---|
| `data/ads/ads_publicados.csv` / `ads_nuevos.csv` | 804 posts de Instagram de 13 cuentas (Kalley, Haceb, Challenger, Mabe, LG, Samsung, TCL, Black+Decker, Imusa, Alkomprar, Alkosto, Ktronix, Falabella) con likes y comentarios reales; impresiones y alcance **estimados** |
| `data/ads/productos_alkosto.csv` | 1.371 productos de alkosto.com: precio, descuento, calificación y número de opiniones (proxy de ventas) |
| `data/imagenes/` | 992 imágenes etiquetadas para entrenar la CNN (Meta Ad Library + Instagram) y una muestra de 10 del test oculto |
| `data/textos/mensajes_train.csv` / `mensajes_muestra_test.csv` | 1.047 textos etiquetados (Meta Ad Library + Instagram) para la LSTM y una muestra de 10 del test oculto |
| `ejemplos_para_subir/imagenes/` | 9 imágenes para probar el simulador |

## Cómo se construyeron

Los scripts de `scripts/` documentan la recolección y el armado:
- **Biblioteca de Anuncios de Meta:** anuncios activos en Colombia de cuentas de marcas y tiendas. Solo muestra anuncios comerciales activos.
- **Instagram:** publicaciones visibles de cada cuenta (texto, likes, comentarios, fecha e imágenes). Se etiquetaron a mano el tipo de imagen y el tipo de mensaje, y se descartó lo que no era de electrodomésticos o tecnología.
- **alkosto.com:** listado de productos por categoría y marca. La calificación y el número de opiniones vienen del widget de reseñas.
- **Armado y validación:** `scripts/construir_dataset_ads.py` arma la base, estima impresiones con benchmarks de alcance orgánico y corre los chequeos de sentido común. `scripts/tendencias.py` resume las tendencias por tipo de imagen y de mensaje.

Los tests de imágenes y textos son **ocultos**: el profesor los califica corriendo los modelos guardados en `modelos/entrenados/`. Los datos crudos (`data_reales/`) y los tests (`_solo_profesor/`) están excluidos de git.

## Aviso sobre los datos

- **Origen:** todos los datos se recolectaron de fuentes **públicas**:
  - publicaciones visibles sin iniciar sesión en Instagram;
  - la Biblioteca de Anuncios de Meta;
  - el catálogo de alkosto.com.
- **Derechos:** las imágenes, textos, logos y marcas pertenecen a sus respectivos dueños (Kalley, Haceb, Challenger, Mabe, LG, Samsung, TCL, Black+Decker, Imusa, Alkomprar, Alkosto, Ktronix, Falabella y demás). Se incluyen únicamente con **fines académicos y no comerciales**, como material de un curso universitario.
- **Sin afiliación:** este proyecto no tiene relación con ninguna de esas marcas ni cuenta con su respaldo.
- **Estimaciones:** las impresiones, el alcance y las métricas derivadas son **estimaciones** construidas para el ejercicio. No son datos reales de las marcas ni deben usarse para evaluar su desempeño.
- **Uso:** no redistribuya este material con fines comerciales.
- **Retiro de contenido:** si usted es titular de algún contenido y desea que se retire, abra un *issue* en este repositorio y se eliminará.
