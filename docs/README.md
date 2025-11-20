# Proyecto OCR clásico (examen IA)

Este proyecto implementa un OCR sencillo **sin usar librerías OCR externas** (no Tesseract, no EasyOCR), solo con visión por computador clásica y un clasificador entrenado sobre conjuntos de caracteres.

## Funcionalidades principales

- Entrada: imagen (PNG/JPG, etc.) con texto.
- OCR de:
  - Texto **impreso** (tipografía de imprenta).
  - Texto **manuscrito** (plantillas rellenadas por compañeros).
- Salida: archivo `.txt` con el texto reconocido.
- Extras opcionales:
  - Detección de **imágenes embebidas** y guardado como archivos independientes.
  - Detección de **tablas** (se generan plantillas Markdown asociadas).
  - Detección de **marcas/códigos** (posibles códigos de barras/QR, sin interpretar).

## Estructura básica

- `main.py`: punto de entrada por línea de comandos.
- `config.py`: parámetros globales (caracteres soportados, rutas, tamaños, etc.).
- `core/`:
  - `preprocessing.py`: carga y binarización de imagen, normalización de caracteres.
  - `segmentation.py`: segmentación en líneas/palabras/caracteres mediante componentes conectados.
  - `features.py`: extracción de descriptores HOG sencillos para cada carácter.
  - `classifier.py`: envoltorio del modelo (KNN/SVM) y utilidades de entrenamiento.
  - `postprocess.py`: reconstrucción de texto a partir de la posición relativa de caracteres.
- `training/`:
  - `generate_synthetic.py`: genera dataset sintético de caracteres impresos renderizando fuentes.
  - `build_handwritten.py`: convierte hojas manuscritas escaneadas en dataset de caracteres.
  - `train_models.py`: entrena dos modelos (impreso/manuscrito) a partir de los datasets.
  - `eval_models.py`: utilidades para inspeccionar los datasets.
- `extras/`:
  - `detect_images.py`: detecta regiones grandes no textuales (imágenes) y las recorta.
  - `detect_tables.py`: intenta detectar tablas mediante morfología de líneas.
  - `detect_markers.py`: detecta regiones con alta densidad de bordes (posibles códigos) y las recorta.
- `models/`: se guardan aquí los modelos entrenados (`printed_ocr_model.pkl`, `handwritten_ocr_model.pkl`).
- `data/`: fuentes, hojas manuscritas, datasets procesados.
- `output/`: resultados (`.txt`, imágenes recortadas, tablas, etc.).

## Flujo típico de uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Preparar fuentes para el dataset impreso

Copiar fuentes `.ttf` / `.otf` a:

```text
data/fonts/
```

### 3. Generar dataset sintético (impreso)

```bash
python -m training.generate_synthetic --fonts-dir data/fonts --out data/processed/printed_synthetic.npz
```

### 4. Preparar hojas manuscritas

1. Diseñar una hoja de plantilla con una cuadrícula (por ejemplo 8x8).
2. Asumimos que `HANDWRITTEN_CHARS` (en `config.py`) se coloca por filas/columnas en esa cuadrícula (A, B, C, ..., 0, 1, etc.).
3. Pedir a los compañeros que rellenen varias copias escribiendo en cada celda el carácter correspondiente múltiples veces.
4. Escanear las hojas a imágenes (`.png` / `.jpg`) y guardarlas en:

```text
data/raw_handwritten/
```

### 5. Construir dataset manuscrito

```bash
python -m training.build_handwritten --sheets-dir data/raw_handwritten --out data/processed/handwritten_dataset.npz --rows 8 --cols 8
```

(Ajustar `--rows` y `--cols` según el diseño real de la plantilla).

### 6. Entrenar modelos

```bash
python -m training.train_models \
  --printed-dataset data/processed/printed_synthetic.npz \
  --handwritten-dataset data/processed/handwritten_dataset.npz \
  --model-type knn
```

Esto crea:

- `models/printed_ocr_model.pkl`
- `models/handwritten_ocr_model.pkl`

### 7. Ejecutar OCR sobre una imagen

#### Texto impreso

```bash
python main.py --mode printed --input data/samples/mi_imagen_impresa.png --out output/mi_imagen_impresa_ocr.txt
```

#### Texto manuscrito

```bash
python main.py --mode handwritten --input data/samples/mi_imagen_manuscrita.png --out output/mi_imagen_manuscrita_ocr.txt
```

### 8. Activar extras opcionales

```bash
python main.py --mode printed --input data/samples/doc_completo.png --detect-images --detect-tables --detect-markers
```

- Las imágenes embebidas se guardan en `output/<nombre>_images/`.
- Las tablas detectadas se guardan en `output/<nombre>_tables/`.
- Las marcas/códigos detectados se guardan en `output/<nombre>_markers/`.

## Notas y restricciones

- No se usa ninguna librería que resuelva OCR completa (no Tesseract, no EasyOCR, etc.).
- Sí se usan:
  - `opencv` para visión clásica (binarización, componentes conectados, morfología…).
  - `numpy` para manejo numérico.
  - `scikit-learn` para el clasificador (KNN o SVM lineal).
  - `Pillow` solo para renderizar texto sintético al generar el dataset impreso.
- El rendimiento dependerá de:
  - Calidad de las imágenes.
  - Calidad y variedad del dataset de entrenamiento.
  - Diseño de la plantilla manuscrita y de las hojas recogidas.

Esta base cubre todo el pipeline lógico y la estructura del proyecto. A partir de aquí se pueden:
- Ajustar hiperparámetros.
- Afinar la segmentación.
- Añadir post-procesado (corrección ortográfica, heurísticas de espacios, etc.).
