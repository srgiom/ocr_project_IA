from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from config import OUTPUT_DIR, PRINTED_MODEL_PATH
from core.segmentation import segment_characters_from_binary, CharSegment
from core.features import extract_features_for_char_images
from core.classifier import OCRModel
from core.postprocess import rebuild_text_from_segments


# -------------------------
#  UTILIDADES DE REJILLA
# -------------------------


def _detect_table_mask(gray: np.ndarray) -> np.ndarray:
    """
    Detecta líneas horizontales y verticales típicas de tablas.
    Devuelve una máscara binaria con las líneas.
    """
    # Binarizar (líneas y texto = 255, fondo = 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = bw.shape

    # Tamaños de kernel adaptados al tamaño de la tabla.
    # Usamos divisores más grandes (30) para no comernos líneas en tablas pequeñas.
    horiz_kernel_size = max(w // 30, 10)
    vert_kernel_size = max(h // 30, 10)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_size, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_size))

    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    table_mask = cv2.add(horiz, vert)
    return table_mask


def _find_grid_lines(mask: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    A partir de una máscara de líneas de tabla, devuelve posiciones de filas y columnas.
    row_lines: coordenadas Y
    col_lines: coordenadas X
    """
    h, w = mask.shape
    _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Proyección vertical (para líneas horizontales)
    proj_y = np.sum(bin_mask > 0, axis=1)
    thr_y = 0.2 * np.max(proj_y) if np.max(proj_y) > 0 else 0  # umbral más laxo

    row_candidates = [i for i, v in enumerate(proj_y) if v >= thr_y]

    row_lines: List[int] = []
    if row_candidates:
        group = [row_candidates[0]]
        for idx in row_candidates[1:]:
            if idx == group[-1] + 1:
                group.append(idx)
            else:
                row_lines.append(int(np.mean(group)))
                group = [idx]
        row_lines.append(int(np.mean(group)))

    # Proyección horizontal (para líneas verticales)
    proj_x = np.sum(bin_mask > 0, axis=0)
    thr_x = 0.2 * np.max(proj_x) if np.max(proj_x) > 0 else 0  # umbral más laxo

    col_candidates = [j for j, v in enumerate(proj_x) if v >= thr_x]

    col_lines: List[int] = []
    if col_candidates:
        group = [col_candidates[0]]
        for idx in col_candidates[1:]:
            if idx == group[-1] + 1:
                group.append(idx)
            else:
                col_lines.append(int(np.mean(group)))
                group = [idx]
        col_lines.append(int(np.mean(group)))

    return row_lines, col_lines


def _split_table_into_cells(table_bgr: np.ndarray) -> List[List[np.ndarray]]:
    """
    Divide una tabla en celdas usando las líneas detectadas.
    Devuelve una lista de filas, cada una con una lista de imágenes de celda.
    """
    gray = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
    mask = _detect_table_mask(gray)

    row_lines, col_lines = _find_grid_lines(mask)
    row_lines = sorted(row_lines)
    col_lines = sorted(col_lines)

    if len(row_lines) < 2 or len(col_lines) < 2:
        # No hay rejilla clara: no dividimos en celdas
        return []

    h, w = gray.shape

    # Aseguramos incluir bordes extremos
    if row_lines[0] > 0:
        row_lines = [0] + row_lines
    if row_lines[-1] < h - 1:
        row_lines = row_lines + [h - 1]

    if col_lines[0] > 0:
        col_lines = [0] + col_lines
    if col_lines[-1] < w - 1:
        col_lines = col_lines + [w - 1]

    cells: List[List[np.ndarray]] = []

    for r in range(len(row_lines) - 1):
        y0 = row_lines[r]
        y1 = row_lines[r + 1]
        # recorte interior para evitar solapar la línea
        y0i = max(0, y0 + 3)
        y1i = min(h, y1 - 3)
        if y1i - y0i < 15:
            continue

        row_cells: List[np.ndarray] = []

        for c in range(len(col_lines) - 1):
            x0 = col_lines[c]
            x1 = col_lines[c + 1]
            x0i = max(0, x0 + 3)
            x1i = min(w, x1 - 3)
            if x1i - x0i < 15:
                continue

            cell = table_bgr[y0i:y1i, x0i:x1i]
            row_cells.append(cell)

        if row_cells:
            cells.append(row_cells)

    return cells


# -------------------------
#  OCR POR CELDA
# -------------------------


def _preprocess_cell_for_ocr(cell_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocesado específico seguro para tablas:
    - Suavizado ligero
    - Binarización OTSU limpia
    - Eliminación suavizada de líneas
    - Nada de escalados agresivos
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    # Suavizado ligero que no destruye texto
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarización OTSU (mucho más estable para dígitos limpios)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Quitar líneas solo si son gruesas (OPEN pequeño)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return clean



def _ocr_cell(cell_bgr: np.ndarray, model: OCRModel) -> str:
    """
    Aplica el OCR clásico a una celda de tabla y devuelve el texto.
    """
    binary = _preprocess_cell_for_ocr(cell_bgr)

    segments: List[CharSegment] = segment_characters_from_binary(binary)
    if not segments:
        return ""

    char_images = [seg.char_image for seg in segments]
    X = extract_features_for_char_images(char_images)
    labels = model.predict(X)

    text = rebuild_text_from_segments(segments, labels)
    # Para tablas: unificamos saltos de línea en espacios
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())

    # Pequeña limpieza de caracteres extraños
    text = text.replace("\t", " ")
    return text.strip()


def _write_markdown_table(md_path: Path, table_cells: List[List[str]]) -> None:
    """
    Escribe una tabla Markdown a partir del contenido OCR de cada celda.
    Si falla la rejilla, se escribe una plantilla simple.
    """
    if not table_cells:
        md_path.write_text(
            "# Tabla detectada (plantilla Markdown)\n\n"
            "No se ha podido extraer una rejilla de celdas de la tabla.\n",
            encoding="utf-8",
        )
        return

    # Normalizar número de columnas (rellenar con cadenas vacías)
    n_cols = max(len(row) for row in table_cells)
    norm_rows = [row + [""] * (n_cols - len(row)) for row in table_cells]

    lines = ["# Tabla detectada (OCR automático)", ""]

    header = norm_rows[0]
    header = [h if h != "" else " " for h in header]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * n_cols) + " |")

    for row in norm_rows[1:]:
        row_clean = [cell if cell != "" else " " for cell in row]
        lines.append("| " + " | ".join(row_clean) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------
#  FUNCIÓN PRINCIPAL
# -------------------------


def detect_and_export_tables(image_bgr: np.ndarray, base_name: str, enable_ocr: bool = True) -> None:
    """
    Detecta tablas simples y las exporta como:
      - imagen recortada (.png)
      - tabla en Markdown (.md)

    Si enable_ocr es True y existe un modelo impreso, intenta hacer OCR por celdas.
    """
    # Si la imagen es muy pequeña, la escalamos para facilitar la detección de rejilla
    h0, w0 = image_bgr.shape[:2]
    min_dim = min(h0, w0)
    if min_dim < 400:
        scale = 400.0 / float(min_dim)
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    table_mask = _detect_table_mask(gray)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables_dir = OUTPUT_DIR / f"{base_name}_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ocr_model: Optional[OCRModel] = None
    if enable_ocr and PRINTED_MODEL_PATH.exists():
        try:
            ocr_model = OCRModel.load(PRINTED_MODEL_PATH)
            print(f"[detect_tables] Modelo impreso cargado desde {PRINTED_MODEL_PATH}")
        except Exception as e:
            print(f"[detect_tables] No se pudo cargar el modelo impreso: {e}")
            ocr_model = None
    elif enable_ocr:
        print(f"[detect_tables] Aviso: no se encuentra modelo impreso en {PRINTED_MODEL_PATH}, tablas sin OCR.")

    idx = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtrar regiones muy pequeñas (ruido)
        if w < 60 or h < 60:
            continue

        crop = image_bgr[y : y + h, x : x + w]
        img_path = tables_dir / f"{base_name}_table_{idx:02d}.png"
        cv2.imwrite(str(img_path), crop)

        md_path = tables_dir / f"{base_name}_table_{idx:02d}.md"

        if ocr_model is not None:
            # 1) Dividir en celdas
            cells_images = _split_table_into_cells(crop)

            # 2) OCR por celda
            table_text_cells: List[List[str]] = []
            for row_cells in cells_images:
                row_texts = [_ocr_cell(cell, ocr_model) for cell in row_cells]
                table_text_cells.append(row_texts)

            # 3) Escribir Markdown de la tabla
            _write_markdown_table(md_path, table_text_cells)
        else:
            # Plantilla vacía si no hay modelo
            md_path.write_text(
                "# Tabla detectada (plantilla Markdown)\n\n"
                "Aquí podría volcarse el contenido OCR de cada celda.\n",
                encoding="utf-8",
            )

        idx += 1

    if idx:
        print(f"[detect_tables] Detectadas {idx} posibles tablas, salidas en: {tables_dir}")
    else:
        print("[detect_tables] No se han detectado tablas claras.")
