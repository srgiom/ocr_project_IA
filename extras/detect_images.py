from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from config import OUTPUT_DIR


def detect_and_save_images(image_bgr: np.ndarray, base_name: str) -> List[Tuple[int, int, int, int]]:
    """Detecta regiones grandes no textuales y las guarda como imágenes independientes.

    Estrategia sencilla:
      - Convertir a gris.
      - Binarizar (texto oscuro sobre fondo claro).
      - Detectar regiones con baja densidad de 'texto'.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar contornos grandes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    images_dir = OUTPUT_DIR / f"{base_name}_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    h_img, w_img = gray.shape[:2]

    idx = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 0.1 * w_img * h_img:
            # Nos quedamos solo con bloques bastante grandes, candidatos a imagen/tabla
            continue

        # Extraer región original a color
        crop = image_bgr[y : y + h, x : x + w]
        out_path = images_dir / f"{base_name}_img_{idx:02d}.png"
        cv2.imwrite(str(out_path), crop)
        regions.append((x, y, w, h))
        idx += 1

    if regions:
        print(f"[detect_images] Guardadas {len(regions)} posibles imágenes en: {images_dir}")
    else:
        print("[detect_images] No se han encontrado regiones grandes tipo imagen.")

    return regions
