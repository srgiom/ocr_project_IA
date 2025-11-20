# Proyecto OCR clásico – Examen Final IA

Este proyecto implementa un sistema OCR propio **sin usar librerías OCR externas** (Tesseract, EasyOCR, etc.).  

Se basa en visión por computador clásica (OpenCV) + descriptores HOG + clasificadores KNN/SVM entrenados sobre datasets de caracteres impresos y manuscritos (sintéticos y/o reales).

---

## 1. Objetivo y relación con el enunciado

### Funcionalidades obligatorias

- ✅ OCR para **texto impreso** (tipografía de imprenta) → texto digital (`--mode printed`).
- ✅ OCR para **texto manuscrito** (escritura manual) → texto digital (`--mode handwritten`), con dos opciones:
  - Dataset manuscrito **sintético** (fuente manuscrita).
  - Dataset manuscrito **real** a partir de hojas plantilla escaneadas.

### Funcionalidades opcionales implementadas

- ✅ **Identificar imágenes embebidas** y guardarlas como archivos independientes  
  → `--detect-images` → `output/<base>_images/`.
- ✅ **Identificar tablas** y generar una representación tipo tabla  
  → `--detect-tables` → `output/<base>_tables/` (imagen + plantilla `.md`).
- ✅ **Detectar marcas/tags/códigos (1D/2D)** y guardarlos aparte, sin interpretar  
  → `--detect-markers` → `output/<base>_markers/`.

> La opción opcional de “detectar características en imágenes para clasificarlas (caras, manos, etc.)” **no se ha implementado**, dado que las anteriores ya son suficientes como extras.

### Restricciones del enunciado

- ❌ No se utiliza ningún motor OCR externo (Tesseract, EasyOCR…).
- ✅ Sí se utilizan:
  - `opencv-python`: operaciones de imagen, binarización, componentes conectados, morfología…
  - `numpy`: operaciones numéricas.
  - `Pillow`: generación de datasets sintéticos e imágenes de prueba.
  - `scikit-learn`: modelo de clasificación (KNN / LinearSVC).

---

## 2. Requisitos de entrada (definidos para este proyecto)

Para que el OCR funcione correctamente, se establecen las siguientes condiciones de entrada:

- **Formato de imagen:**  
  - Imágenes raster: `PNG`, `JPG`, `BMP`, `TIFF`… (cualquier formato soportado por OpenCV `cv2.imread`).

- **Aspecto general:**
  - Fondo claro (blanco o casi blanco).
  - Texto oscuro (negro o casi negro).
  - Texto con orientación horizontal global (inclinación menor de ±10°).

- **Texto impreso (`--mode printed`):**
  - Fuentes limpias (Arial / FreeSans / similares).
  - Tamaño de letra aproximado entre 24–120 px en la imagen.
  - Sin columnas múltiples muy complejas.

- **Texto manuscrito (`--mode handwritten`):**
  - Idealmente mayúsculas + dígitos (caracteres definidos en `HANDWRITTEN_CHARS` en `config.py`).
  - Trazos dentro del área central (sin tocar bordes de la imagen).
  - Se recomienda escribir sobre fondo blanco uniforme.

---

## 3. Estructura del proyecto

```text
ocr_project/
  main.py
  config.py
  requirements.txt

  core/
    __init__.py
    preprocessing.py      # binarización y normalización de caracteres
    segmentation.py       # segmentación en líneas/palabras/caracteres
    features.py           # HOG sencillo
    classifier.py         # modelos KNN / LinearSVC + wrapper OCRModel
    postprocess.py        # reconstrucción de texto (líneas/palabras)

  training/
    generate_synthetic.py             # dataset impreso sintético
    generate_synthetic_handwritten.py # dataset manuscrito sintético
    build_handwritten.py              # dataset manuscrito real (hojas escaneadas)
    train_models.py                   # entrena modelos impreso + manuscrito
    eval_models.py                    # utilidades para inspeccionar datasets

  extras/
    detect_images.py   # detecta grandes regiones tipo imagen
    detect_tables.py   # detecta tablas y genera plantilla Markdown
    detect_markers.py  # detecta marcas/códigos y los recorta

  data/
    fonts/            # fuentes .ttf/.otf/.ttc (impreso + manuscrito)
    raw_handwritten/  # hojas manuscritas escaneadas (para dataset real)
    processed/        # datasets .npz generados
    samples/          # imágenes de prueba

  models/
    printed_ocr_model.pkl      # modelo impreso entrenado
    handwritten_ocr_model.pkl  # modelo manuscrito entrenado

  output/
    ...             # textos OCR, imágenes recortadas, tablas, marcas...

  docs/
    README.md       # documentación del proyecto
````

Además, en la raíz hay scripts auxiliares:

* `generate_test_image.py`
  → Genera imágenes de **texto impreso** para probar el OCR.
* `generate_handwritten_template.py`
  → Genera una hoja plantilla para recoger escritura manuscrita real.
* `generate_handwritten_test_image.py`
  → Genera imágenes de **texto manuscrito sintético** (para pruebas).

---

## 4. Puesta en marcha desde cero

### 4.1. Crear entorno virtual e instalar dependencias

```bash
cd ocr_project

python3 -m venv venv
source venv/bin/activate   # en macOS / Linux
# .\venv\Scripts\activate  # en Windows

pip install -r requirements.txt
```

> Se ha probado con Python 3.9+, OpenCV, scikit-learn, Pillow, numpy.

---

## 5. Preparar fuentes

Debes colocar al menos:

* 1 fuente **impresa** (Arial, FreeSans, etc.).
* 1 fuente **manuscrita** (Patrick Hand, Caveat, etc.).

Ejemplo:

```text
data/fonts/FreeSans.ttf
data/fonts/PatrickHand-Regular.ttf
```

En macOS puedes copiar, por ejemplo, Arial:

```bash
cp "/System/Library/Fonts/Arial.ttf" data/fonts/
```

Y una manuscrita descargada (por ejemplo de Google Fonts) en `data/fonts/`.

---

## 6. Generar datasets de entrenamiento

### 6.1. Dataset impreso sintético

Usa fuentes “normales” (Arial / FreeSans). Genera muchas muestras artificialmente:

```bash
python -m training.generate_synthetic \
  --fonts-dir data/fonts \
  --out data/processed/printed_synthetic.npz \
  --samples-per-char 50
```

Resultado:

* `data/processed/printed_synthetic.npz` con `X` (features HOG) y `y` (caracteres impresos).

---

### 6.2. Dataset manuscrito sintético (Opción B, rápida)

Se usa una fuente manuscrita (por ejemplo `PatrickHand-Regular.ttf`) para simular escritura a mano:

```bash
python -m training.generate_synthetic_handwritten \
  --font data/fonts/PatrickHand-Regular.ttf \
  --out data/processed/handwritten_synthetic.npz \
  --samples-per-char 80
```

Resultado:

* `data/processed/handwritten_synthetic.npz` con `X` y `y` manuscritos sintéticos.

> Esta opción basta para tener un modelo manuscrito funcional.
> Si además quieres dataset manuscrito real, ver sección 7.

---

## 7. (Opcional) Dataset manuscrito real desde hojas escaneadas

### 7.1. Generar plantilla de recogida

```bash
python generate_handwritten_template.py \
  --out data/raw_handwritten/plantilla_handwritten.png \
  --rows 8 \
  --cols 8
```

1. Imprimir `plantilla_handwritten.png`.
2. Pedir a compañeros que escriban varias veces el carácter indicado en cada celda.
3. Escanear las hojas resultantes y guardarlas en `data/raw_handwritten/`:

```text
data/raw_handwritten/hoja1.png
data/raw_handwritten/hoja2.png
...
```

### 7.2. Construir dataset manuscrito real

```bash
python -m training.build_handwritten \
  --sheets-dir data/raw_handwritten \
  --out data/processed/handwritten_dataset.npz \
  --rows 8 \
  --cols 8
```

> `rows` y `cols` deben coincidir con los usados al generar la plantilla.

Luego podrás elegir en el entrenamiento si usar el sintético, el real, o fusionarlos (editando rutas).

---

## 8. Entrenamiento de modelos

Con los datasets ya generados:

```bash
python -m training.train_models \
  --printed-dataset data/processed/printed_synthetic.npz \
  --handwritten-dataset data/processed/handwritten_synthetic.npz \
  --model-type knn
```

Esto crea:

* `models/printed_ocr_model.pkl`
* `models/handwritten_ocr_model.pkl`

Si no tienes dataset manuscrito aún, el script avisa y entrena solo el modelo impreso.

---

## 9. Generar imágenes de prueba

### 9.1. Texto impreso de prueba

```bash
python generate_test_image.py \
  --text "HOLA ESTO ES UNA PRUEBA DE OCR IMPRESO" \
  --font data/fonts/FreeSans.ttf \
  --out data/samples/test_ocr_1.png \
  --width 1200 \
  --height 400 \
  --font-size 72
```

### 9.2. Texto manuscrito sintético de prueba

```bash
python generate_handwritten_test_image.py \
  --text "Hola esto es una prueba de OCR manuscrito" \
  --font data/fonts/PatrickHand-Regular.ttf \
  --out data/samples/test_handwritten1.png \
  --width 1200 \
  --height 450 \
  --font-size 90
```

---

## 10. Ejecución del OCR

### 10.1. OCR texto impreso

```bash
python main.py \
  --mode printed \
  --input data/samples/test_ocr_1.png \
  --out output/test_ocr_1.txt
```

Salida:

* `output/test_ocr_1.txt`

### 10.2. OCR texto manuscrito

```bash
python main.py \
  --mode handwritten \
  --input data/samples/test_handwritten1.png \
  --out output/handwritten1.txt
```

Salida:

* `output/handwritten1.txt`

---

## 11. Activar funcionalidades opcionales

Sobre cualquier imagen:

```bash
python main.py \
  --mode printed \
  --input data/samples/doc_completo.png \
  --out output/doc_completo_ocr.txt \
  --detect-images \
  --detect-tables \
  --detect-markers
```

* `--detect-images`
  → `output/doc_completo_images/` con las regiones grandes tipo imagen.

* `--detect-tables`
  → `output/doc_completo_tables/` con:

    * `doc_completo_table_XX.png` (imagen de tabla).
    * `doc_completo_table_XX.md` (plantilla Markdown de tabla asociada).

* `--detect-markers`
  → `output/doc_completo_markers/` con regiones que parecen códigos / marcas.

---

## 12. Resumen del pipeline interno

1. **Preprocesado (`core/preprocessing.py`)**

    * Conversión a gris.
    * Suavizado.
    * Binarización adaptativa (texto = blanco, fondo = negro en binaria).
    * Pequeña morfología para limpiar ruido.

2. **Segmentación (`core/segmentation.py`)**

    * Componentes conectados para detectar blobs.
    * Agrupación en líneas según posición vertical.
    * División en palabras según gaps horizontales.
    * Normalización de cada carácter a tamaño fijo (`CHAR_SIZE`).

3. **Características (`core/features.py`)**

    * HOG casero sobre cada parche de carácter.
    * Vector final fijo por carácter.

4. **Clasificación (`core/classifier.py`)**

    * KNN / LinearSVC entrenado sobre datasets sintéticos/ reales.
    * `OCRModel` mapea índices → caracteres reales.

5. **Reconstrucción (`core/postprocess.py`)**

    * Reordenación por `line_idx`, `word_idx`, `char_idx`.
    * Construcción de líneas y espacios.
    * Generación del texto final.

---

## 13. Limitaciones conocidas

* Manuscrito muy irregular, con letras muy pegadas o caligrafías muy artísticas, puede degradar el rendimiento.
* No hay corrector ortográfico ni modelo de lenguaje; las correcciones son puramente visuales.
* Las tablas se detectan y se les asocia una plantilla Markdown, pero no se realiza OCR por celda en esta versión.

---

Con este README y los scripts asociados, el proyecto se puede ejecutar **desde cero** siguiendo únicamente los comandos indicados, y cumple los requisitos del enunciado (obligatorios + varias funcionalidades opcionales) respetando la restricción de no usar librerías OCR externas.

```
```
