# Parcial – Simulador de Ads con Machine Learning

**Curso:** Estadística 3 / Big Data – Machine Learning
**Profesor:** Jose Daniel Ramirez Soto
**Tema:** Clasificación de imágenes (CNN), clasificación de texto (LSTM y regresión logística) y modelos de árboles para tomar una decisión de negocio con **datos reales**.
**Fecha de entrega:** viernes 4 de diciembre de 2026, 11:59 pm (una semana). La presentación y la sustentación son parte de la entrega.

---

## 1. Contexto

Las marcas de electrodomésticos en Colombia (Kalley, Haceb, Challenger, Mabe, Alkosto, Ktronix…) publican todos los días en Instagram y pautan anuncios en Facebook/Instagram. Antes de pagar la pauta, el equipo de mercadeo quiere responder una sola pregunta:

> **¿Esta publicación va a funcionar mejor que lo normal para la marca?**

Para decidir solo puede usar lo que conoce **antes de publicar**:

1. **La imagen**: ¿muestra **personas**, el **producto** o un **descuento**?
2. **El mensaje**: ¿habla de una **promoción** (marca, tienda, regalo, crédito), de las **características** del producto o del **descuento**?
3. **El producto y el momento**: categoría, precio, calificación del producto en la tienda, fecha y hora de publicación, si es carrusel, si es un concurso…

Ya existe un **simulador en Streamlit** (`app.py`) que hace exactamente ese flujo, pero hoy funciona con **reglas hechas a mano** (mock). **Su trabajo es entrenar los tres modelos con datos reales, evaluarlos bien y conectarlos al simulador.**

```
 imagen ──► CNN ─────────────────► tipo_imagen ──┐
 texto  ──► LSTM (y Reg. logística) ► tipo_mensaje ┼──► Random Forest ──► P(tasa de interacción > promedio) ──► ¿publicar?
 producto + fecha + formato ───────────────────────┘
```

> **Los datos son reales.** 804 publicaciones de Instagram de 13 cuentas colombianas de electrodomésticos y tecnología (Kalley, Haceb, Challenger, Mabe, LG, Samsung, TCL, Black+Decker, Imusa, Alkomprar, Alkosto, Ktronix y Falabella), con sus likes y comentarios reales; anuncios pagados de la [Biblioteca de Anuncios de Meta](https://www.facebook.com/ads/library/); y 1.371 productos de alkosto.com con precio, descuento, calificación y número de opiniones. Lo único estimado son las **impresiones** y el **alcance**, que Instagram no publica (sección 2.2).

---

## 2. Teoría: métricas de publicidad digital y cómo definimos la etiqueta

### 2.1 El embudo

```
Impresiones ──► Clicks / Interacciones ──► Conversiones (ventas)
   (vieron)          (se interesaron)            (compraron)
```

| Métrica | Fórmula | Qué mide |
|---|---|---|
| **Impresiones** | veces que el anuncio apareció en pantalla | alcance |
| **CTR** (*click-through rate*) | clicks / impresiones | qué tan atractivo es el anuncio |
| **Tasa de conversión (CR)** | conversiones / clicks | qué tan bien convierte el interés en ventas |
| **CPM** | costo / impresiones × 1.000 | costo de alcance |
| **CPC** | costo / clicks | costo de cada visita |
| **CPA** | costo / conversiones | costo de cada venta |
| **ROAS** | ingresos / costo de la pauta | retorno de la inversión publicitaria |
| **Tasa de interacción** (*engagement rate*) | (likes + comentarios + …) / seguidores (o / alcance) | qué tanto reacciona la audiencia |

Un CTR típico en Facebook/Instagram está alrededor de 1%; una tasa de interacción típica de una marca grande en Instagram está por debajo de 1% de sus seguidores. Comparar números absolutos entre marcas no sirve: una cuenta de 500.000 seguidores siempre tendrá más likes que una de 40.000.

### 2.2 Los proxies que usamos

Las impresiones, los clicks y las ventas de un anuncio son **privados** (solo los ve la marca en su administrador de anuncios). Lo que sí es público en Instagram son los **likes**, los **comentarios** y los **seguidores**. Por eso usamos:

| Lo que queremos medir | Proxy observable | Variable |
|---|---|---|
| CTR (interés) | (likes + comentarios) / seguidores | `tasa_interaccion` |
| **Ventas** | **likes** | `ventas_proxy` (= `likes`) |
| Tasa de conversión | likes / impresiones estimadas | `tasa_conversion_proxy` |
| Impresiones y alcance | **estimados** con benchmarks de alcance orgánico (Instagram no los publica) | `impresiones_estimadas`, `alcance_estimado` |
| Ventas del producto en la tienda | número de opiniones en alkosto.com | `ventas_proxy` en `productos_alkosto.csv`; `num_resenas` en la base |

**Cómo se estiman las impresiones.** Según [Socialinsider](https://socialinsider.io/blog/instagram-reels-statistics/), una imagen de marca llega a ~7% de los seguidores en cuentas pequeñas y el porcentaje baja con el tamaño de la cuenta; los carruseles llegan más lejos. Usamos:

```
tasa_alcance = 7% × (seguidores / 3.000)^-0,15        (≈4% con 100 mil, ≈3% con 500 mil)
alcance      = seguidores × tasa_alcance × 1,25 si es carrusel × (tasa_post / tasa_promedio_cuenta)^0,5 × ruido
impresiones  = alcance × U(1,15–1,40)
```

La interacción sobre alcance queda alrededor de 1%, en el rango que reportan [Dash Social](https://www.dashsocial.com/social-media-benchmarks/instagram) y [Buffer](https://buffer.com/resources/instagram-engagement-rate/).

### 2.3 La etiqueta

```
poner_ad = 1   si   tasa_interaccion del post  >  tasa_interaccion PROMEDIO de su cuenta
poner_ad = 0   en otro caso
```

Se compara contra el promedio **de la misma cuenta** para que la etiqueta mida si el post funcionó **mejor de lo normal para esa marca**, y no si la marca es grande.

⚠️ La tasa de interacción tiene una distribución muy sesgada (unos pocos posts virales, sobre todo concursos, suben mucho el promedio). Por eso menos de la mitad de los posts quedan por encima del promedio: **las clases están desbalanceadas** y debe reportar F1, recall y AUC, no solo accuracy.

⚠️ **Fuga de información.** `likes`, `comentarios`, `interacciones` y `tasa_interaccion` son el **resultado** del post. Nunca pueden ser variables del modelo.

### 2.4 Hipótesis para contrastar con los datos

| # | Hipótesis | Fuente |
|---|---|---|
| H1 | En Colombia los mensajes e imágenes de **descuento** funcionan mejor, sobre todo en temporadas de ofertas (Black Days) | [EY 2025](https://www.ey.com/es_co/newsroom/2025/10/adn-del-consumidor-colombiano-demuestra-priorizacion-por-precio): el precio es el criterio principal de compra; [Amazon/Mobile Time](https://mobiletime.la/?p=822347): 77% compra más en periodos promocionales |
| H2 | Las imágenes con **personas** generan más interacción que las de solo producto | [Agorapulse](https://agorapulse.com/social-media-lab/should-your-twitter-ads-contain-human-faces-or-graphics), [Columbia](https://www.columbia.edu/~on2110/Papers/AI_in_disguise.pdf) (con evidencia en contra: [Adobe](https://business.adobe.com/uk/blog/perspectives/about-face-why-you-should-stop-showing-faces-in-your-ads)) |
| H3 | En productos **caros** (alto involucramiento) funcionan mejor los mensajes de **características**; en productos baratos, el descuento | Modelo de probabilidad de elaboración ([revisión crítica](https://researchgate.net/profile/Gayle-Kerr/publication/277681837_The_elaboration_likelihood_model_Review_critique_and_research_agenda/links/57d8ee1708ae6399a3993a9a/The-elaboration-likelihood-model-Review-critique-and-research-agenda.pdf)); [La República](https://www.larepublica.co/globoeconomia/el-comprador-que-compra-electrodomesticos-ahora-esta-mas-informado-2911252): los electrodomésticos son lo que más se investiga antes de comprar |
| H4 | Los **concursos** ("comenta y gana") disparan los comentarios | — (verifíquelo) |
| H5 | Los productos **populares** y bien calificados en la tienda atraen más interacción (prueba social) | — (verifíquelo) |

---

## 3. Estructura del proyecto

```
src/bigdata_parcial/
├── ENUNCIADO.md · README.md · requirements.txt
├── app.py                         <- simulador (NO es necesario modificarlo)
├── modelos/
│   ├── imagenes.py                <- mock de reglas de color + cargador de la CNN
│   ├── mensaje.py                 <- mock de palabras clave + cargadores de LSTM y regresión logística
│   ├── clicks.py                  <- mock de reglas + cargador del Random Forest
│   ├── variables.py               <- construye las variables del modelo (las MISMAS en el notebook y en el simulador)
│   ├── rutas.py                   <- nombres de archivo que el simulador busca
│   └── entrenados/                <- AQUÍ guarda sus modelos
├── data/
│   ├── imagenes/train/<clase>/    <- imágenes reales etiquetadas (personas / producto / descuento)
│   ├── imagenes/muestra_test/     <- 10 imágenes de muestra del test oculto (etiquetas en muestra_test_etiquetas.csv)
│   ├── textos/mensajes_train.csv  <- textos etiquetados (para entrenar)
│   ├── textos/mensajes_muestra_test.csv <- 10 textos de muestra del test oculto, con etiqueta
│   ├── ads/ads_publicados.csv     <- posts reales con resultado (likes, comentarios, poner_ad)
│   ├── ads/ads_nuevos.csv         <- posts reales SIN resultado (para predecir)
│   ├── ads/productos_alkosto.csv  <- 1.371 productos de alkosto.com (precio, calificación, ventas_proxy)
│   └── ads/imagenes_posts/        <- imágenes de cada post
├── ejemplos_para_subir/imagenes/  <- imágenes para probar el simulador
├── notebooks/plantilla_parcial.ipynb
├── scripts/                       <- cómo se recolectaron y armaron los datos
└── entrega/                       <- AQUÍ deja sus predicciones y el slide
```

---

## 4. Diccionario de datos

### 4.1 `data/imagenes/` – imágenes reales (224×224)

| Clase | Criterio |
|---|---|
| `personas` | una o más personas son protagonistas |
| `producto` | el electrodoméstico es protagonista y no hay un precio o % grande |
| `descuento` | el precio, el % de descuento o el evento de ofertas es lo principal |

| Clase | Train | Test oculto (lo tiene el profesor) |
|---|---|---|
| `personas` | 354 | 86 |
| `producto` | 362 | 86 |
| `descuento` | 276 | 66 |

Imágenes de la Meta Ad Library y portadas de Instagram (en los reels, la portada del video). Varias imágenes son el mismo creativo con otro recorte: el test se armó **agrupando las imágenes casi iguales**.

**El test es oculto.** Usted solo recibe una **muestra de 10 imágenes** (`data/imagenes/muestra_test/`, con etiquetas) para verificar que su flujo de predicción funciona. El profesor califica **corriendo su modelo guardado** sobre el test completo. Un modelo que solo memorizó train va a perder puntos en el test.

### 4.2 `data/textos/mensajes_train.csv` – textos reales

| Columna | Descripción |
|---|---|
| `fuente` | `meta_ad_library` (anuncio pagado) o `instagram` (publicación) |
| `id`, `cuenta` | identificador y marca |
| `texto` | texto tal como se publicó |
| `tipo_mensaje` | **objetivo**: `promocion`, `caracteristicas`, `descuento` |

- `descuento`: el mensaje principal es el precio, el % o "antes/ahora".
- `caracteristicas`: describe especificaciones o beneficios del producto.
- `promocion`: marca, tienda, evento, concurso, crédito, regalo u obsequio, sin una rebaja concreta.

- **`mensajes_train.csv`**: 1.047 textos etiquetados (518 de características, 313 de promoción y 216 de descuento; 827 de Instagram y 220 de la Ad Library).
- **Test oculto**: 217 anuncios de la Meta Ad Library que tiene el profesor. Los textos repetidos o casi iguales (misma plantilla) quedaron todos del mismo lado. Usted recibe una **muestra de 10** con etiqueta en `mensajes_muestra_test.csv`.
- Los textos se copiaron tal como venían de las fuentes (hashtags, emojis, menciones, enlaces). Decida qué limpieza aplica y justifíquela.

En Instagram el texto que se clasifica es el **primer comentario del post**, es decir, el caption que escribe la marca.

### 4.2b `data/ads/productos_alkosto.csv` – catálogo real de la tienda

`sku`, `nombre`, `marca`, `categoria`, `precio_antes`, `precio`, `descuento_pct`, `calificacion`, `ventas_proxy` (número de opiniones), `url`. Sirve para explorar qué productos y marcas "venden" más por categoría (TV, celulares, computadores, audio, neveras, lavadoras, freidoras…).

### 4.3 `data/ads/ads_publicados.csv` y `ads_nuevos.csv` – publicaciones reales de Instagram

| Columna | Tipo | Descripción |
|---|---|---|
| `post_id`, `url`, `fecha`, `texto`, `imagenes` | id | identificación del post (no son variables) |
| `cuenta`, `seguidores` | cat / num | marca y seguidores de la cuenta |
| `producto_mencionado`, `menciona_producto` | cat / num | producto detectado en el texto (freidora, nevecon, celular…) y si el post nombra algún producto |
| `tasa_historica_categoria` | num | tasa de interacción promedio de los posts **anteriores** de la misma cuenta y categoría (proxy del "promedio de clicks por producto"; no usa el propio post ni posts futuros) |
| `es_video` | 0/1 | el post es un reel (la imagen es la portada) |
| `tipo_imagen` | cat | tipo de la imagen de portada → en el simulador lo da **su CNN** |
| `tipo_mensaje` | cat | tipo del texto → en el simulador lo da **su LSTM** |
| `categoria_producto` | cat | detectada del texto (nevera, lavadora, televisor, …, `marca_institucional`) |
| `temporada`, `dia_semana`, `hora_publicacion` | cat / num | momento de publicación |
| `es_carrusel`, `n_imagenes` | num | formato |
| `precio_referencia`, `calificacion_producto` | num | precio mediano y calificación de esa marca y categoría en alkosto.com (**tienen nulos**) |
| `num_resenas` | num | reseñas de esa marca y categoría en alkosto.com. **Solo para exploración (H5): no es variable del modelo** ni está en `FEATURES` |
| `num_palabras`, `num_hashtags`, `num_emojis`, `menciona_precio`, `descuento_mencionado`, `tiene_pregunta`, `es_concurso`, `llamado_accion` | num | variables calculadas del texto (ver `modelos/variables.py`) |
| `likes`, `comentarios`, `interacciones`, `tasa_interaccion`, `tasa_promedio_cuenta`, `alcance_estimado`, `tasa_interaccion_alcance`, `ventas_proxy`, `tasa_conversion_proxy` | num | ⚠️ **resultado: NO usar como variables** |
| `poner_ad` | 0/1 | **objetivo de clasificación** (sección 2.3) |
| `likes` (= ventas proxy), `impresiones_estimadas` | num | resultado del post (solo para explorar; **no** son variables del modelo) |

Los posts institucionales (fechas especiales, marca, tiendas) no nombran producto: por eso `precio_referencia` y `calificacion_producto` (y `num_resenas`) son nulos en ~59% de los posts. Decida cómo imputarlos (pista: `menciona_producto`).

---

## 5. Lo que debe hacer

Use `notebooks/plantilla_parcial.ipynb` como guía. Fije una semilla (`random_state`).

### Parte 0 – Métricas y etiqueta (3%)

1. Explique con sus palabras CTR, tasa de conversión y tasa de interacción, y por qué usamos proxies.
2. Muestre la distribución de `tasa_interaccion` por cuenta (escala logarítmica) y la proporción de `poner_ad = 1`. ¿Por qué no es 50%? ¿Qué pasaría si la etiqueta usara la mediana?

### Parte A – CNN: tipo de imagen (10%)

1. Cargue `data/imagenes/train/`, muestre ejemplos y el conteo por clase.
2. Separe train / validación estratificado (la validación sale de train).
3. **Líneas base:** clase mayoritaria y regresión logística sobre píxeles (accuracy, F1 macro, matriz de confusión).
4. **CNN:** al menos 2 bloques `Conv2D + MaxPooling` y `class_weight`. Curvas de *loss* y *accuracy*. *(Opcional: transfer learning con `MobileNetV2`.)*
5. **Data augmentation:** con ~1.000 imágenes la CNN memoriza train rápido. Agregue capas de aumento **dentro del modelo** (`RandomRotation`, `RandomZoom`, `RandomContrast`, `RandomBrightness` y ruido con `GaussianNoise`) y compare las curvas de validación **con y sin** aumento.
6. Matriz de confusión, F1 macro, `classification_report`. ¿Qué clases se confunden?
7. Prediga las 10 imágenes de `data/imagenes/muestra_test/` y compárelas con `muestra_test_etiquetas.csv`. **Reporte el F1 macro de validación de su modelo final:** se compara con el del test oculto.

📚 [`src/notebook/CNN.ipynb`](../notebook/CNN.ipynb), [`src/notebook_class/supervisado/supervisado_neural_nets.ipynb`](../notebook_class/supervisado/supervisado_neural_nets.ipynb).

### Parte B – Tipo de mensaje: LSTM y regresión logística (10%)

1. **Exploración**: distribución de clases por fuente y por marca, longitud de los textos, palabras más frecuentes por clase.
2. **Muestreo** (dentro de `mensajes_train.csv`): split **por marca** (`GroupShuffleSplit`, `groups=cuenta`) y split aleatorio estratificado. Compare los resultados: cada marca repite sus plantillas de texto, ¿cuál split es más honesto?
3. **Línea base – regresión logística** (`TfidfVectorizer` + `LogisticRegression`) con las 10 palabras más importantes por clase. **Esta es la ventaja de un modelo explicable:** los coeficientes muestran qué términos empujan cada clase. Revíselos con ojo crítico: ¿tienen sentido para decidir el tipo de mensaje? ¿Hay términos que solo identifican a la marca o a la fuente y que convendría quitar?
4. **LSTM**: `TextVectorization` → `Embedding` → `LSTM` → `Dense(3, softmax)`. Curvas de entrenamiento.
5. Matrices de confusión y F1 macro de ambos modelos. ¿La LSTM le gana a la regresión logística con tan pocos datos? ¿Cuál pondría en producción?
6. Prediga los 10 textos de `mensajes_muestra_test.csv` con los dos modelos y **reporte el F1 macro de validación** del modelo final. Ojo: el test oculto son anuncios pagados y la mayoría de train son publicaciones de Instagram. ¿Cambia la distribución de clases? ¿Cómo lo afecta?

📚 [`src/notebook_class/supervisado/supervisado_neural_nets.ipynb`](../notebook_class/supervisado/supervisado_neural_nets.ipynb), [`src/notebook_class/supervisado/supervisado_regresion_logistic.ipynb`](../notebook_class/supervisado/supervisado_regresion_logistic.ipynb), [`src/notebook/logistic_regression_by_hand.ipynb`](../notebook/logistic_regression_by_hand.ipynb).

### Parte C – ¿Publicamos? Random Forest (14%)

1. **Exploración y prueba de hipótesis H1–H5**: tasa de éxito por `tipo_imagen × tipo_mensaje`, por categoría, por hora y día, efecto de `es_concurso`, relación con `num_resenas`. ¿Qué hipótesis se confirman con estos datos y cuáles no?
2. Use **solo** las variables de `modelos.variables.FEATURES` (las mismas que usa el simulador).
3. **Muestreo**: split estratificado 75/25 y, como comparación, un **split temporal** (entrenar con los posts más antiguos, probar con los más recientes). ¿Cambian los resultados?
4. **Línea base**: regresión logística (imputación + one-hot + estandarización en un `Pipeline`).
5. **Random Forest** con `class_weight="balanced"` y búsqueda de hiperparámetros (`GridSearchCV`) **solo sobre train**.
6. En test: matriz de confusión, accuracy, precision, recall, F1, AUC y curva ROC de ambos modelos.
7. **Feature importance** agrupada por variable original. ¿Qué controla mercadeo y qué no?
8. Prediga `data/ads/ads_nuevos.csv` → `entrega/predicciones_ads_nuevos.csv` (`post_id,p_exito,poner_ad`). El profesor también corre su modelo guardado sobre estos posts.

📚 [`src/notebook_class/supervisado/arboles.ipynb`](../notebook_class/supervisado/arboles.ipynb), [`src/notebook/Trees.ipynb`](../notebook/Trees.ipynb), [`src/notebook_class/supervisado/supervisado_datos.ipynb`](../notebook_class/supervisado/supervisado_datos.ipynb), [`src/notebook_class/exploracion/exploracion_datos.ipynb`](../notebook_class/exploracion/exploracion_datos.ipynb).

### Parte D – Simulador (3%)

1. `streamlit run app.py`. Guarde sus modelos en `modelos/entrenados/` con los nombres de la sección 6 y presione **🔄 Recargar modelos**: deben quedar los tres en 🟢.
2. Pruebe una imagen de `ejemplos_para_subir/imagenes/` con un texto real y responda con la simulación imagen × mensaje y la curva por hora: *¿qué recomendaría publicar para una nevera de Kalley? ¿y para una freidora de aire en Black Days?*

### Parte E – Presentación y sustentación (60%)

Prepare una presentación de máximo **12 diapositivas** (`entrega/presentacion.pdf`) para el gerente de mercadeo. Es la base de su sustentación. Debe incluir:

1. **Problema y datos**: qué se quiere decidir, de dónde salen los datos, las **métricas** (CTR, conversión, tasa de interacción, proxies) y cómo se define la etiqueta `poner_ad`.
2. **Exploración**: los hallazgos principales de cada dataset, con gráficas:
   - imágenes: conteo por clase y ejemplos;
   - mensajes: distribución por clase, fuente y marca, longitud de los textos y palabras más frecuentes por clase;
   - posts: `tasa_interaccion` por cuenta, tasa de éxito por `tipo_imagen × tipo_mensaje`, por categoría y por hora, y el resultado de las hipótesis H1–H5.
3. **Muestreo**: cómo dividió cada dataset (estratificación, grupos por marca, imágenes casi iguales, split temporal), con la distribución de clases en train y validación, y cómo evitó la fuga de información.
4. **Matrices de confusión** de **todos** los modelos, cada una comparada con su línea base:
   - CNN vs. clase mayoritaria y regresión logística sobre píxeles;
   - LSTM vs. regresión logística de mensajes (con las palabras más importantes por clase y lo que limpió del texto);
   - Random Forest vs. regresión logística para `poner_ad` (con la curva ROC).
5. **Tabla resumen de métricas** (en validación):

   | Modelo | Métricas mínimas |
   |---|---|
   | CNN y modelos de mensajes | accuracy, **F1 macro**, precision y recall por clase |
   | Clasificador `poner_ad` | accuracy, precision, recall, **F1**, **AUC** |

   Explique qué métrica usó para escoger cada modelo y por qué (clases desbalanceadas, costo de publicar un mal ad vs. dejar de publicar uno bueno).
6. **Feature importance** del Random Forest: ¿qué controla mercadeo y qué no?
7. **Simulador** con los modelos en 🟢 y recomendaciones concretas para mercadeo.
8. **Limitaciones**: proxies, posts orgánicos vs. pauta, tamaño de muestra, sobreajuste.

---

## 6. Contrato de los modelos

| Modelo | Archivo(s) en `modelos/entrenados/` | Formato |
|---|---|---|
| CNN | `cnn_imagenes.keras` (o `.h5`) + `cnn_config.json` | softmax de 3 clases. Config: `{"img_size": [64, 64], "color": "rgb", "clases": ["personas","producto","descuento"], "normalizar": true}` (`false` si la división por 255 está dentro del modelo). |
| Mensaje – LSTM | `mensaje_lstm.keras` (o carpeta `mensaje_lstm/`) + `mensaje_lstm_config.json` | Modelo keras que recibe **texto crudo** (`TextVectorization` adentro). Config: `{"clases": ["promocion","caracteristicas","descuento"], "normalizar_texto": false}` |
| Mensaje – Reg. logística | `mensaje_lr.joblib` | `Pipeline([("vectorizer", TfidfVectorizer(...)), ("clf", LogisticRegression(...))])` |
| Decisión | `ads_random_forest.joblib` | `Pipeline([("pre", ColumnTransformer(...)), ("rf", RandomForestClassifier(...))])` que recibe un DataFrame con las columnas de `modelos.variables.FEATURES`; el último paso debe tener `feature_importances_`. |

---

## 7. Entrega

Cree una rama en su repositorio, agregue la carpeta `parcial` con el proyecto resuelto y abra un *pull request* sobre `master` con **jdramirez** como *reviewer* (otros medios: −0.5).

- [ ] Notebook(s) ejecutados con todas las salidas.
- [ ] `modelos/entrenados/` con los modelos y configs.
- [ ] `entrega/predicciones_ads_nuevos.csv`, `entrega/presentacion.pdf`.
- [ ] En el notebook, el F1 macro de **validación** de su CNN y de su modelo de mensajes final.
- [ ] El simulador corre con `streamlit run app.py` y muestra 🟢 en los tres modelos.

## 8. Calificación

| Criterio | Peso |
|---|---|
| 0 – Métricas y etiqueta | 3% |
| A – CNN (notebook + desempeño en el test oculto) | 10% |
| B – Mensajes: LSTM + regresión logística (notebook + desempeño en el test oculto) | 10% |
| C – Random Forest: exploración, hipótesis, muestreo, clasificación (`poner_ad`), feature importance | 14% |
| D – Simulador conectado | 3% |
| **E – Presentación y sustentación**: exploración, matrices de confusión, métricas, interpretación y respuestas en vivo | **60%** |
| Bonos (se miden **corriendo sus modelos** de `modelos/entrenados/` sobre los tests ocultos): mejor AUC en `ads_nuevos` (+0.2), mejor F1 macro en el test oculto de imágenes (+0.2), mejor F1 macro en el test oculto de mensajes (+0.2) | extra |

**Calificación con test oculto:** las partes A y B se califican con el modelo que usted guardó, no con un archivo de predicciones. Si el modelo no carga con el contrato de la sección 6, la parte no tiene nota de desempeño.

**Sobreajuste:** si su modelo rinde en el test oculto bastante por debajo de lo que reporta en validación, pierde hasta un **20% de la parte**. Valide con un split honesto (por marca en textos) y use *data augmentation* en imágenes.

**Sustentación:** cada estudiante explica en vivo por qué su modelo toma una decisión concreta (una imagen, un texto o un post que escoge el profesor): palabras y coeficientes, matriz de confusión, importancia de variables, efecto de cambiar una variable en el simulador. Lo que no pueda explicar no se califica como propio.

**Penalizaciones:** usar `likes/comentarios/interacciones/tasa_interaccion` como variables (Parte C = 0); ajustar hiperparámetros con el test (−30% de la parte); notebook sin ejecutar o no reproducible (−20%).
