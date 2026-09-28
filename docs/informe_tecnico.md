# Informe técnico: clasificador barco / no barco para el UAV de inspección portuaria

Inteligencia Artificial, Ingeniería Mecatrónica, periodo 2026-2. Proyecto 2 (segundo corte). Este documento es la evidencia E2 del formato ABET y describe también el protocolo de la prueba en vivo (E3 y E4).

## 1. Planteamiento

El dron recibe recortes de 80×80 píxeles en RGB y tiene que decidir si en cada uno hay un barco. Dahana y Gurning (2020) proponen la vigilancia aérea precisamente porque el AIS y el VTS muestran la posición de un buque como un ícono, sin la imagen real; el clasificador es la parte que hace esa confirmación visual a bordo, como en las pruebas de inspección con drones del puerto de Rotterdam.

La meta de la asignatura es un accuracy mayor al 98 % sobre un conjunto de prueba que no se conoce de antemano, con una penalización de 0.5 por cada 2 % por debajo. Por eso el trabajo tuvo dos partes. La primera fue optimizar el clasificador sobre ShipsNet con validación cruzada. La segunda, comprobar que el modelo sigue funcionando con imágenes de otros satélites, otras resoluciones y otros puertos, porque nada garantiza que la prueba venga del mismo sensor que el dataset.

## 2. Datos

| Conjunto | Uso | Imágenes | Con barco | Origen |
|---|---|---:|---:|---|
| ShipsNet | entrenamiento y validación cruzada | 4000 | 1000 | Kaggle, PlanetScope a 3 m (bahías de San Francisco y San Pedro) |
| Puertos Maxar | entrenamiento | 1528 | 513 | Maxar Open Data: Valencia, Colombo, Tampa, Kingston (dos fechas), Iskenderun, Durban y Ravenna, remuestreadas a 1.2 m |
| Escenas Planet | prueba externa | 201 | 39 | escenas completas que trae el mismo dataset de Kaggle, recortes nuevos |
| Rotterdam | prueba externa | 322 | 57 | SpaceNet 6, WorldView-2 a 0.5 m llevado a un mosaico de 1 m |

En ShipsNet la clase "barco" es una imagen centrada en un único barco completo. La clase "no barco" incluye agua, tierra, muelles, barcos parciales cortados por el borde y confusores que otros modelos marcaban como barco. Los conjuntos nuevos siguen la misma regla: los positivos tienen el barco centrado y completo, ocupando entre el 45 % y el 95 % del lado del recorte; los negativos son ventanas al azar (agua, muelles, grúas, contenedores, zonas urbanas) y barcos cortados por el borde.

Las etiquetas nuevas se hicieron a mano. En Rotterdam los candidatos los propuso un detector YOLOv8 público entrenado con imágenes de Google Earth y cada recorte se revisó a ojo; los dudosos (40) se descartaron. En las escenas de Maxar los barcos se marcaron sobre la imagen con una rejilla de coordenadas y los negativos se revisaron uno por uno para quitar los que tenían algún bote. Las coordenadas de todos los recortes están en `datos_externos/*.json` y `entrenamiento/armar_externos.py` los reconstruye desde las fuentes públicas (se verificó que el resultado es idéntico píxel a píxel).

Las escenas de Planet no son del todo independientes de ShipsNet: son las mismas zonas, y para 28 de los 39 positivos existe en ShipsNet un recorte casi idéntico (similitud coseno mayor a 0.95 entre las imágenes reducidas a 20×20 en gris, probando las 8 orientaciones). Rotterdam sí es independiente: otro sensor, otra resolución, otro puerto y otra época.

## 3. Protocolo de validación

- Validación cruzada estratificada de 5 folds sobre ShipsNet, semilla 42. Todos los modelos (clásicos y redes) usan exactamente los mismos cortes, así que las comparaciones son pareadas.
- Métricas: accuracy, precisión, recall y F1 de la clase barco, y AUC.
- En los recortes de Maxar hay varios recortes del mismo barco a distintas escalas. Para que un mismo barco no quede a la vez en entrenamiento y validación se usa `StratifiedGroupKFold` agrupando por barco.
- Los conjuntos de prueba externa nunca entran al entrenamiento de los modelos que se evalúan. Cada uno de los 5 modelos de la validación cruzada los predice y se reporta el promedio.
- El modelo que usa la interfaz se reentrena al final con todos los datos (incluido Rotterdam). Las cifras de este informe vienen de los modelos de validación, que no vieron los datos con los que se miden.

## 4. Línea base

Antes de optimizar se fijó un punto de partida sin descriptores: cada imagen reducida a 40×40×3 y estandarizada, con regresión logística (C = 0.01). Da 93.00 % ± 0.26 de accuracy. Un KNN (k = 5) sobre las 50 primeras componentes principales llega a 93.23 % ± 0.57. Como referencia, decir "no barco" a todo ya da 75 %, así que la línea base aprende algo, pero está lejos de la meta y su recall de barcos es bajo (F1 de 0.86).

## 5. Preprocesamiento y descriptores visuales

Se probaron tres descriptores, cada uno con el mismo SVM de kernel RBF (C = 10, γ = scale) para que la comparación dependa solo del descriptor:

- HOG sobre la imagen en gris (celdas de 8 px, 9 orientaciones, bloques de 2×2 con normalización L2-Hys, 2916 valores). Describe la silueta alargada del casco y sus bordes.
- LBP uniforme (P = 8, R = 1): histograma de toda la imagen más uno por cuadrante, 50 valores. Describe textura.
- Color: histogramas HSV de 16 niveles más media y desviación de cada canal RGB, 54 valores.

{{tabla_clasicos}}

El HOG solo ya pasa de 93 % a 98.87 %. El color por sí mismo no separa las clases (77.85 %) porque el agua y los barcos cambian de tono entre escenas, y el LBP aporta poco (94.63 %). Juntar los tres descriptores sube apenas a 98.90 %, y un Random Forest sobre el mismo vector queda en 96.72 %.

Para calibrar el SVM se hizo una búsqueda en malla de C ∈ {1, 3, 10, 30, 100} y γ ∈ {scale, 10⁻⁴, 3·10⁻⁴, 10⁻³} con validación cruzada anidada (3 folds internos dentro de cada uno de los 5 externos). La superficie es plana para C ≥ 3 y el único valor claramente malo es γ = 10⁻³, que sobreajusta y cae a 92 % (figura 1). La malla no mejora el valor por defecto: el SVM ya estaba en su techo con estos descriptores.

![Sensibilidad del SVM](figuras/sensibilidad_svm.png)

*Figura 1. Accuracy de la validación interna en función de C y γ. Las curvas de γ = scale y γ = 3·10⁻⁴ casi coinciden.*

{{hog_sensibilidad}}

Con descriptores clásicos el modelo queda en 98.9 % con una desviación de 0.4 entre folds. Es un margen muy corto sobre el 98 % para confiar en un conjunto de prueba desconocido, y eso motivó pasar a redes convolucionales.

{{secciones_redes}}

## 10. Modelo final

La red elegida es una ResNet18-D preentrenada en ImageNet (pesos de la librería timm), ajustada con ShipsNet, los puertos de Maxar y Rotterdam: 5850 imágenes, 1570 con barco. Se entrenó 12 épocas con AdamW (tasa máxima 10⁻³ con política one-cycle, decaimiento de pesos 10⁻⁴), lotes de 64, suavizado de etiquetas de 0.05 y peso de la clase barco igual a la razón negativos/positivos para compensar el desbalance. El aumento de datos combina las 8 orientaciones de 90° y reflejo, rotación libre de ±45°, escala entre 0.85 y 1.2, desplazamiento, cambios de color (saturación, balance de blancos, contraste, brillo y gamma), desenfoque, pérdida de resolución y ruido.

El modelo se exportó a ONNX y la interfaz lo corre con ONNX Runtime, así que el computador de la prueba (o el del dron) no necesita PyTorch. En la inferencia se promedian las 8 orientaciones de cada imagen (TTA) y se usa umbral 0.5 sobre p(barco). Antes de exportar se verificó que ONNX y PyTorch dan la misma salida (diferencia máxima {{dif_onnx}}).

{{rendimiento_final}}

## 11. Prueba en vivo (E3 y E4)

La interfaz (`app.py`) sigue este flujo:

1. Se cargan las imágenes de la carpeta de prueba (por defecto `test/`; también se puede escoger otra o arrastrar archivos). Las imágenes que no son de 80×80 se reescalan y la interfaz lo avisa.
2. El modelo clasifica todo el lote al cargar. Con la opción "Ocultar la predicción hasta etiquetar" activada, quien etiqueta no ve la salida del modelo antes de decidir.
3. Cada imagen se etiqueta con los botones o con las teclas `b` y `n`. Si los archivos ya traen la etiqueta en el nombre (formato de ShipsNet) o en subcarpetas `barco/` y `no_barco/`, se pueden usar directamente.
4. Con cada etiqueta se recalculan accuracy, precisión, recall y F1, la matriz de confusión y la curva de accuracy acumulado. El accuracy lleva un intervalo de confianza de Wilson al 95 %.
5. Para el contraste con la validación cruzada la interfaz muestra una tabla con ambas columnas y revisa si el accuracy de la validación cruzada cae dentro del intervalo de confianza en vivo. Si queda por encima, hay indicios de que las imágenes de prueba difieren de las de entrenamiento (sensor, escala o criterio de etiquetado) y la pestaña de errores permite ver cuáles fallaron.
6. "Guardar evaluación" deja en `resultados_en_vivo/` un CSV con la predicción, la probabilidad y la etiqueta de cada imagen, y un JSON con la matriz de confusión y las métricas.

El intervalo de confianza importa para interpretar el resultado. Con 100 imágenes y 98 aciertos, el intervalo de Wilson va de 93.0 % a 99.4 %, así que una diferencia de uno o dos puntos frente a la validación cruzada no alcanza para decir que el modelo empeoró.

## 12. Limitaciones

- La definición de clase pesa mucho. En ShipsNet un barco cortado por el borde cuenta como "no barco"; si en la prueba se etiqueta como "barco" cualquier imagen donde se vea parte de un barco, el modelo va a discrepar en esos casos por diseño.
- Las etiquetas de Rotterdam y de Maxar las hizo una sola persona. En Rotterdam quedaron casos discutibles (pontones y muelles flotantes, filas de barcazas amarradas) que explican buena parte de los errores restantes.
- Los botes pequeños (lanchas, veleros de marina) no se incluyeron como barco en los datos nuevos, siguiendo ShipsNet, que está hecho para buques.

## Referencias

- Dahana, U. y Gurning, R. O. S. (2020). Maritime Aerial Surveillance: Integration Manual Identification System to Automatic Identification System. IOP Conference Series: Earth and Environmental Science, 557, 012014.
- Hammell, R. Ships in Satellite Imagery (ShipsNet). Kaggle: https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery
- The Maritime Executive. Rotterdam tests drones for ship inspections and monitoring the port: https://maritime-executive.com/article/rotterdam-tests-drones-for-ship-inspections-and-monitoring-the-port
- Shermeyer, J. et al. (2020). SpaceNet 6: Multi-Sensor All Weather Mapping Dataset. CVPR Workshops.
- Maxar Open Data Program, catálogo de eventos: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
- Dalal, N. y Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR.
- Ojala, T., Pietikäinen, M. y Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI, 24(7).
- He, K., Zhang, X., Ren, S. y Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
- He, T. et al. (2019). Bag of Tricks for Image Classification with Convolutional Neural Networks. CVPR (variante ResNet-D).
- Wightman, R. PyTorch Image Models (timm): https://github.com/huggingface/pytorch-image-models
