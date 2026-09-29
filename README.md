# Clasificador barco / no barco para el UAV de inspección portuaria

Proyecto 2 de Inteligencia Artificial (Ingeniería Mecatrónica, 2026-2). El clasificador recibe recortes aéreos o satelitales de 80×80 píxeles en RGB y decide si hay un barco centrado en la imagen. Está pensado como la parte de visión del sistema de percepción de un dron que vigila el tráfico de un puerto, con el puerto de Rotterdam como caso de estudio.

El repositorio trae la interfaz para la prueba en vivo, el modelo ya entrenado y todo el código de los experimentos. El detalle de la metodología y los resultados está en [docs/informe_tecnico.md](docs/informe_tecnico.md).

## Resultados principales

Validación cruzada estratificada de 5 folds sobre ShipsNet (4000 imágenes), con los mismos cortes para todos los modelos:

| Configuración | Accuracy |
|---|---:|
| Línea base: píxeles + regresión logística | 93.00 % |
| HOG + color + LBP + SVM (búsqueda en malla de C y γ) | 98.90 % |
| Lo mismo con celdas HOG de 16 px (según el análisis de sensibilidad) | 99.33 % |
| CNN propia sin aumento de datos | 98.92 % |
| ResNet18-D entrenada desde cero | 97.93 % |
| ResNet18-D preentrenada en ImageNet, TTA | 99.88 % |
| **Modelo final**: ResNet18-D + puertos Maxar, TTA | **99.88 %** |

Prueba externa del modelo final, con los modelos de validación (no vieron estas imágenes): 99.6 % en recortes nuevos de las escenas de Planet y 94.0 % en el puerto de Rotterdam (SpaceNet 6, otro satélite y otra resolución), donde el mismo modelo entrenado solo con ShipsNet se quedaba en 84.5 % y reconocía apenas el 12 % de los barcos.

![Interfaz](docs/figuras/interfaz.png)

## Cómo correr la interfaz

Se necesita Python 3.10 o más reciente. La interfaz solo depende de Streamlit y ONNX Runtime; no hace falta instalar PyTorch. Todo se corre desde la carpeta raíz del repositorio, la que tiene `app.py`.

En una terminal:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Desde Visual Studio Code:

1. Abrir la carpeta del repositorio (Archivo → Abrir carpeta).
2. `Ctrl+Shift+P` → **Python: Create Environment** → **Venv**, y marcar solo el `requirements.txt` de la raíz (el de `entrenamiento/` es para reentrenar y trae PyTorch). Si Pylance sigue diciendo que no encuentra `streamlit` o `altair`, elegir ese entorno (`.venv`) con **Python: Select Interpreter**.
3. Abrir `app.py` y ejecutarlo con el botón ▶. El script se relanza solo como `streamlit run app.py`.

La interfaz se abre en el navegador (http://localhost:8501) y se detiene con `Ctrl+C` en la terminal. Para la prueba:

1. Copiar las imágenes que entregue el docente en la carpeta `test/` (o en cualquier otra y escribir la ruta, o usar **Elegir carpeta…**). También se pueden arrastrar los archivos al recuadro de la barra lateral.
2. Pulsar **Cargar carpeta**. El modelo clasifica todas las imágenes de una vez; las que no son de 80×80 se reescalan.
3. Etiquetar cada imagen con **Barco** / **No barco** (o las teclas `b` y `n`). Por defecto la predicción del modelo queda oculta hasta poner la etiqueta, para no sesgar a quien etiqueta.
4. Las métricas se actualizan con cada etiqueta: accuracy con su intervalo de confianza del 95 %, precisión, recall, F1, matriz de confusión y la curva de accuracy acumulado. Al lado aparece la validación cruzada del modelo para contrastar.
5. **Guardar evaluación** deja un CSV (imagen por imagen) y un JSON con el resumen en `resultados_en_vivo/`.

Para tener un enlace público se puede desplegar el repositorio en Streamlit Community Cloud con `app.py` como archivo principal. En ese caso la ruta de carpeta no sirve (apunta al servidor) y las imágenes se suben con el recuadro de arrastrar.

Si las imágenes ya vienen etiquetadas en el nombre (formato de ShipsNet, `1__...png` / `0__...png`) o en subcarpetas `barco/` y `no_barco/`, la interfaz ofrece usar esas etiquetas directamente. En `muestras/` hay 30 imágenes de ejemplo recortadas de las escenas de Planet; no están en el conjunto de entrenamiento, aunque algunos de esos barcos también aparecen en ShipsNet.

## Estructura

```
app.py               interfaz (Streamlit)
clasificador.py      carga del modelo ONNX, preprocesamiento y TTA
modelo/              barcos.onnx e info.json (métricas de validación que muestra la interfaz)
entrenamiento/       experimentos, entrenamiento final y exportación
datos_externos/      recortes adicionales (npz) y coordenadas de dónde salió cada uno
resultados/          CSV de validación cruzada, robustez y predicciones fuera de fold
docs/                informe técnico y figuras
test/                carpeta por defecto para la prueba en vivo
muestras/            imágenes de ejemplo
```

## Reentrenar

```bash
pip install -r entrenamiento/requirements.txt
cd entrenamiento
python descargar.py pesos        # pesos ImageNet de timm (releases de GitHub)
python descargar.py shipsnet     # necesita la API de Kaggle; o bajar el zip a mano en data/
python exp_clasicos.py           # línea base y descriptores + SVM (validación cruzada)
python exp_cnn.py cnn_sin_aug cnn_geom cnn_aug resnet18d_scratch resnet18d_pre mnv3_pre
python exp_dominios.py resnet18d_pre
python robustez.py resnet18d_pre cnn_geom cnn_sin_aug
python latencia.py
python entrenar_final.py resnet18d_pre
python figuras.py
python tablas.py                 # tablas del informe en markdown
```

Los recortes de Maxar, Rotterdam y de las escenas de Planet ya vienen en `datos_externos/*.npz`. Si se quieren regenerar desde las fuentes originales está `armar_externos.py`, que usa las coordenadas guardadas en los JSON de esa misma carpeta.

## Datos

- **ShipsNet** ([Kaggle, rhammell](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)): 4000 recortes de PlanetScope a 3 m, 1000 con barco. Es la base del entrenamiento y de la validación cruzada.
- **Maxar Open Data Program** (CC BY-NC 4.0): 9 escenas de 7 puertos (Valencia, Colombo, Tampa, Kingston en dos fechas, Iskenderun, Durban y Ravenna) remuestreadas a 1.2 m. De ahí salieron 1528 recortes, 513 con alguno de los 67 barcos marcados a mano. Se usan para entrenar.
- **SpaceNet 6, Rotterdam** (CC BY-SA 4.0): imágenes WorldView-2 del puerto de Rotterdam. 322 recortes revisados uno por uno, 57 con barco. Son la prueba externa de los modelos de validación; el modelo final, ya evaluado, se reentrenó incluyéndolos.
- **Escenas completas de Planet** que vienen con el dataset de Kaggle: 201 recortes nuevos, solo de prueba.

Los candidatos a barco en Rotterdam se buscaron con un detector YOLOv8 público entrenado con imágenes de Google Earth ([robmarkcole](https://github.com/robmarkcole/kaggle-ships-in-satellite-imagery-with-YOLOv8)); ese detector solo sirvió para proponer recortes y no hace parte del clasificador.
