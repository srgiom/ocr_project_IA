from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from config import OUTPUT_DIR


def _detect_table_mask(gray: np.ndarray) -> np.ndarray:
    """Intenta detectar líneas horizontales/verticales típicas de tablas."""
    # Binarizar
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detectar líneas horizontales
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    # Detectar líneas verticales
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    table_mask = cv2.add(horiz, vert)
    return table_mask


def detect_and_export_tables(image_bgr: np.ndarray, base_name: str) -> None:
    """Detecta tablas simples y las exporta como Markdown básico.

    Versión simplificada:
      - Detecta líneas rectas que formen una rejilla.
      - Extrae bounding boxes de tablas.
      - NO realiza OCR dentro de las celdas (eso quedaría como extensión).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    table_mask = _detect_table_mask(gray)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables_dir = OUTPUT_DIR / f"{base_name}_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 50:
            continue
        crop = image_bgr[y : y + h, x : x + w]
        img_path = tables_dir / f"{base_name}_table_{idx:02d}.png"
        cv2.imwrite(str(img_path), crop)

        # Por ahora, solo guardamos la imagen de la tabla y un .md vacío asociado.
        md_path = tables_dir / f"{base_name}_table_{idx:02d}.md"
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
