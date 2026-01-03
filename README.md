# Proyecto OCR clásico – Examen Final IA

Este proyecto implementa un sistema OCR propio **sin usar librerías OCR externas** (Tesseract, EasyOCR, etc.).

Se basa en visión por computador clásica (OpenCV) + descriptores HOG + clasificadores KNN/SVM entrenados sobre datasets de caracteres impresos y manuscritos (sintéticos y/o reales).

---

## 1. Relación con el enunciado del examen

### Funcionalidades obligatorias

- ✅ OCR para **texto impreso** → texto digital  
  `python main.py --mode printed ...`
- ✅ OCR para **texto manuscrito** → texto digital  
  `python main.py --mode handwritten ...`  

Ambos modos comparten pipeline:

1. Preprocesado (binarización).
2. Segmentación en caracteres.
3. Extracción de HOG.
4. Clasificación con modelo entrenado.
5. Reconstrucción de líneas y palabras.

### Funcionalidades opcionales implementadas

- ✅ Detectar **imágenes embebidas** y guardarlas recortadas  
  `--detect-images` → `output/<base>_images/`
- ✅ Detectar **tablas** y generar:
  - imagen recortada de la tabla
  - **archivo Markdown con la tabla y OCR automático por celda** (cuando es posible)  
  `--detect-tables` → `output/<base>_tables/`
- ✅ Detectar **marcas/códigos** (barras, QR, tags) y guardarlos como imágenes independientes (sin interpretarlos)  
  `--detect-markers` → `output/<base>_markers/`

> No se implementa la clasificación de “caras, manos, paisajes…”, ya que con las extras anteriores se cubre suficientemente la parte opcional.

### Restricciones del ejercicio

- ❌ No se usa ningún motor OCR externo (Tesseract, EasyOCR, etc.).
- ✅ Se usan únicamente:
  - **opencv-python** – operaciones de visión por computador clásica
    (binarización, morfología, componentes conectados y segmentación),
    sin utilizar módulos OCR ni detección de texto de alto nivel.
  - **numpy** – cálculo numérico y gestión de matrices, vectores de características
    y datasets de entrenamiento.
  - **Pillow** – generación de imágenes y datasets sintéticos a partir de fuentes
    tipográficas (`.ttf`), así como creación de imágenes de prueba.
  - **scikit-learn** – implementación de clasificadores clásicos
    (KNN y LinearSVC) entrenados desde cero sobre descriptores HOG.

---

## 2. Requisitos de entrada definidos

Para que el OCR funcione correctamente se aplican las siguientes restricciones:

- **Formato de imagen de entrada**
  - Cualquier formato soportado por OpenCV: `PNG`, `JPG`, `BMP`, `TIFF`, …

- **Condiciones geométricas / visuales**
  - Fondo claro (idealmente blanco casi uniforme).
  - Texto oscuro (negro o muy oscuro).
  - Texto globalmente horizontal (rotación global < ±10°).
  - Sin deformaciones de perspectiva extremas.

- **Texto impreso (`--mode printed`)**
  - Fuentes legibles tipo PatrickHand-Regular, FreeSans, etc.
  - Tamaño de letra en rango aproximado [24, 120] píxeles de altura.

- **Texto manuscrito (`--mode handwritten`)**
  - Trazos dentro de la zona central del documento (sin cortar caracteres en bordes).
  - Fondo blanco (folio escaneado, tablet con fondo claro).

---

## 3. Pipeline interno (resumen técnico)

1. **Preprocesado – `core/preprocessing.py`**

    * Convertir BGR → escala de grises.
    * Suavizado gaussiano.
    * Binarización adaptativa inversa (fondo = 0, texto = 255).
    * Morfología para limpiar ruido.

2. **Segmentación – `core/segmentation.py`**

    * Componentes conectados (`connectedComponentsWithStats`).
    * Agrupación en líneas por coordenada vertical.
    * División en palabras usando gaps horizontales.
    * Recorte de cada carácter y normalización a `CHAR_SIZE`.

3. **Características – `core/features.py`**

    * Gradientes Sobel.
    * HOG casero por celdas y bloques.
    * Vector de características fijo por carácter.

4. **Clasificación – `core/classifier.py`**

    * KNN o LinearSVC sobre vectores HOG.
    * `OCRModel` mapea índices de clase ↔ caracteres (`PRINTED_CHARS` / `HANDWRITTEN_CHARS`).

5. **Reconstrucción – `core/postprocess.py`**

    * Orden por `(line_idx, word_idx, char_idx)`.
    * Inserción de espacios entre palabras y saltos de línea.
    * Generación de texto final en UTF-8.

6. **Extras – `extras/*.py`**

    * `detect_images` → grandes regiones no textuales.
    * `detect_tables` → localización de tablas, recorte, división en celdas y OCR por celda.
    * `detect_markers` → detección de regiones tipo códigos de barras/QR.

---

## 4. Estructura del proyecto

```text
ocr_project/
  main.py
  setup_all.py
  config.py
  requirements.txt

  core/
    __init__.py
    preprocessing.py       # binarización + normalización de caracteres
    segmentation.py        # segmentación en líneas / palabras / caracteres
    features.py            # HOG casero
    classifier.py          # modelos KNN / LinearSVC + wrapper OCRModel
    postprocess.py         # reconstrucción final del texto

  training/
    generate_synthetic.py              # dataset sintético IMPRESO
    generate_synthetic_handwritten.py  # dataset sintético MANUSCRITO
    build_handwritten.py               # dataset manuscrito desde plantillas escaneadas
    build_handwritten_from_images.py   # dataset manuscrito desde imágenes sueltas (A_nombre.png…)
    train_models.py                    # entrena modelos impreso + manuscrito
    eval_models.py                     # utilidades de inspección de datasets

  extras/
    detect_images.py    # detecta grandes regiones no textuales (imágenes)
    detect_tables.py    # detecta tablas y genera imagen + .md con OCR por celda
    detect_markers.py   # detecta marcas/códigos y los recorta

  generators/
    __init__.py
    generate_test_image.py             # genera imágenes de texto impreso para pruebas
    generate_handwritten_test_image.py # genera imágenes de texto manuscrito sintético
    generate_handwritten_template.py   # genera plantilla para recopilar manuscrito real
    generate_table_samples.py          # genera tablas de ejemplo (vacía + tabla del 5)

  data/
    fonts/              # fuentes .ttf/.otf/.ttc (impreso + manuscrito)
    raw_handwritten/
      my_dataset/       # dataset manuscrito real (carpetas Mayusculas/Minusculas/Numeros …)
      ...               # plantillas manuscritas escaneadas (si se usan)
    processed/          # datasets .npz generados (X, y)
    samples/            # imágenes de prueba

  models/
    printed_ocr_model.pkl       # modelo OCR impreso
    handwritten_ocr_model.pkl   # modelo OCR manuscrito

  output/
    *.txt                      # resultados OCR
    *_images/                  # imágenes embebidas recortadas
    *_tables/                  # tablas detectadas + Markdown con OCR por celda
    *_markers/                 # marcas/códigos detectados

````

---

## 5. Puesta en marcha desde cero

### 5.1. Crear entorno virtual e instalar dependencias

```bash
cd ocr_project

python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# .\venv\Scripts\activate       # Windows (PowerShell/CMD)

pip install -r requirements.txt
```

---

## 6. Preparar fuentes

Se necesitan, como mínimo:

* Una fuente **impresa** (p.ej. `FreeSans.ttf`, `Arial.ttf`…).
* Una fuente **manuscrita** (p.ej. `PatrickHand-Regular.ttf`, `Caveat`, etc.).

Colocarlas en:

```text
data/fonts/FreeSans.ttf
data/fonts/PatrickHand-Regular.ttf
```

Ejemplo en macOS para copiar Arial:

```bash
cp "/System/Library/Fonts/Arial.ttf" data/fonts/
```

---

## 7. Setup automático recomendado (`setup_all.py`)

La forma más rápida de dejar el proyecto listo es usar el script de setup:

```bash
python setup_all.py
```

Este script realiza:

1. **Selección de fuentes**

    * Busca fuentes en `data/fonts/`.
    * Elige una como “impresa” y otra como “manuscrita” (por nombre).

2. **Generación de datasets**

    * `data/processed/printed_synthetic.npz`
      Dataset sintético impreso (a partir de fuentes impresas).
    * `data/processed/handwritten_synthetic.npz`
      Dataset sintético manuscrito (a partir de fuente manuscrita).
    * Si existe `data/raw_handwritten/my_dataset/` con imágenes reales:

        * Construye `data/processed/handwritten_from_images.npz` usando
          `training.build_handwritten_from_images`.
        * Fusiona sintético + real → `data/processed/handwritten_merged.npz`.

3. **Entrenamiento de modelos**

    * `models/printed_ocr_model.pkl` (impreso).
    * `models/handwritten_ocr_model.pkl` (manuscrito, usando el dataset fusionado si existe real).

4. **Generación de imágenes de prueba**

    * `data/samples/setup_printed_test.png`
    * `data/samples/setup_handwritten_test.png`

5. **Ejecución de OCR sobre estas imágenes**

    * `output/setup_printed_test.txt`
    * `output/setup_handwritten_test.txt`

Después de ejecutar `setup_all.py` el proyecto queda listo para usar con tus propias imágenes.

---

## 8. Uso manual de datasets (por si se quiere controlar todo a mano)

### 8.1. Dataset impreso sintético

```bash
python -m training.generate_synthetic \
  --fonts-dir data/fonts \
  --out data/processed/printed_synthetic.npz \
  --samples-per-char 50
```

### 8.2. Dataset manuscrito sintético

```bash
python -m training.generate_synthetic_handwritten \
  --font data/fonts/PatrickHand-Regular.ttf \
  --out data/processed/handwritten_synthetic.npz \
  --samples-per-char 80
```

### 8.3. Dataset manuscrito REAL desde plantillas escaneadas (opcional)

1. Generar plantilla:

   ```bash
   python -m generators.generate_handwritten_template \
     --out data/raw_handwritten/plantilla_handwritten.png \
     --rows 8 \
     --cols 8
   ```

2. Imprimir, rellenar a mano, escanear y guardar en `data/raw_handwritten/hoja1.png`, `hoja2.png`, …

3. Construir dataset:

   ```bash
   python -m training.build_handwritten \
     --sheets-dir data/raw_handwritten \
     --out data/processed/handwritten_dataset.npz \
     --rows 8 \
     --cols 8
   ```

### 8.4. Dataset manuscrito REAL desde imágenes sueltas A_nombre.png

Si tienes un dataset manual ya recortado, p.ej.:

```text
data/raw_handwritten/my_dataset/
  Mayusculas/
    A_sergio.png
    B_sergio.png
    ...
  Minusculas/
    a_sergio.png
    ...
  Numeros/
    0_sergio.png
    1_sergio.png
    ...
```

El carácter se extrae del **primer símbolo del nombre** (`A_...`, `7_...` etc.).

Construir dataset:

```bash
python -m training.build_handwritten_from_images \
  --dir data/raw_handwritten/my_dataset \
  --out data/processed/handwritten_from_images.npz
```

### 8.5. Fusión sintético + real (opcional, si se hace manual)

Ejemplo de fusión manual desde Python (si no se usa `setup_all.py`):

```bash
python - << 'EOF'
import numpy as np
from pathlib import Path

base = Path("data/processed")
synth = np.load(base / "handwritten_synthetic.npz")
real  = np.load(base / "handwritten_from_images.npz")

X = np.concatenate([synth["X"], real["X"]], axis=0)
y = np.concatenate([synth["y"], real["y"]], axis=0)

np.savez_compressed(base / "handwritten_merged.npz", X=X, y=y)
print("Fusionado en data/processed/handwritten_merged.npz")
EOF
```

---

## 9. Entrenamiento de modelos

Con los datasets listos (sintético o fusionado):

```bash
python -m training.train_models \
  --printed-dataset data/processed/printed_synthetic.npz \
  --handwritten-dataset data/processed/handwritten_merged.npz \
  --model-type knn
```

o, si solo se usa sintético manuscrito:

```bash
python -m training.train_models \
  --printed-dataset data/processed/printed_synthetic.npz \
  --handwritten-dataset data/processed/handwritten_synthetic.npz \
  --model-type knn
```

Genera:

* `models/printed_ocr_model.pkl`
* `models/handwritten_ocr_model.pkl`

---

## 10. Generar imágenes de prueba

### 10.1. Texto impreso

```bash
python -m generators.generate_test_image \
  --text "HOLA ESTO ES UNA PRUEBA DE OCR IMPRESO" \
  --font data/fonts/FreeSans.ttf \
  --out data/samples/test_ocr_1.png \
  --width 1200 \
  --height 400 \
  --font-size 72
```

### 10.2. Texto manuscrito sintético

```bash
python -m generators.generate_handwritten_test_image \
  --text "Hola esto es una prueba de OCR manuscrito" \
  --font data/fonts/PatrickHand-Regular.ttf \
  --out data/samples/test_handwritten1.png \
  --width 1200 \
  --height 450 \
  --font-size 90
```

### 10.3. Tablas de ejemplo (vacía + tabla del 5)

```bash
python -m generators.generate_table_samples
```

Genera, por ejemplo:

```text
data/samples/tabla_multiplicar_5.png
data/samples/tabla_vacia_5x5.png
```

Estas imágenes son ideales para probar la detección de tablas + OCR por celda.

---

## 11. Ejecución del OCR

### 11.1. OCR texto impreso

```bash
python main.py \
  --mode printed \
  --input data/samples/test_ocr_1.png \
  --out output/test_ocr_1.txt
```

Salida:

* `output/test_ocr_1.txt`

### 11.2. OCR texto manuscrito

```bash
python main.py \
  --mode handwritten \
  --input data/samples/test_handwritten1.png \
  --out output/test_handwritten1.txt
```

Salida:

* `output/test_handwritten1.txt`

---

## 12. Extras opcionales: imágenes, tablas, marcas

### 12.1. Uso combinado de extras (se pueden usar todos a la vez)

```bash
python main.py \
  --mode printed \
  --input data/samples/doc_completo.png \
  --out output/doc_completo_ocr.txt \
  --detect-images \
  --detect-tables \
  --detect-markers
```

Se generan:

* **Imágenes embebidas**
  `output/doc_completo_images/`

* **Tablas**
  `output/doc_completo_tables/` con:

    * `doc_completo_table_XX.png` – imagen de la tabla.
    * `doc_completo_table_XX.md` – tabla Markdown con contenido OCR automático por celda (si la rejilla se detecta correctamente; en caso contrario se deja como plantilla vacía con estructura).

* **Marcas / códigos**
  `output/doc_completo_markers/` con regiones que parecen códigos de barras/QR/tags.

### 12.2. Detección de imágenes embebidas

```bash
python main.py \
  --mode printed \
  --input data/samples/doc_completo.png \
  --out output/doc_completo_ocr.txt \
  --detect-images
```

### 12.3. Detección y OCR de tablas

```bash
python main.py \
  --mode printed \
  --input data/samples/doc_completo.png \
  --out output/doc_completo_ocr.txt \
  --detect-tables
```

### 12.4. Detección de QRs

```bash
python main.py \
  --mode printed \
  --input data/samples/doc_completo.png \
  --out output/doc_completo_ocr.txt \
  --detect-markers
```

---

## 13. Limitaciones conocidas

* Escritura manuscrita muy irregular, con letras muy pegadas o extremadamente estilizadas, puede degradar la tasa de acierto.
* El OCR por celda en tablas funciona muy bien con tablas generadas limpias (como las de `generate_table_samples`), pero en tablas complejas o con bordes poco definidos puede fallar parcialmente o cometer errores de caracter (`1 ↔ I`, `0 ↔ O`, etc.).

---
