from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from config import OUTPUT_DIR


def detect_and_save_markers(image_bgr: np.ndarray, base_name: str) -> List[Tuple[int, int, int, int]]:
    """Detección muy simple de posibles códigos de barras / QR / marcas.

    Estrategia:
      - Trabajar sobre imagen en gris y binaria.
      - Buscar regiones con alta densidad de bordes y aspecto rectangular razonable.
      - Solo se recortan y guardan; no se interpretan.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers_dir = OUTPUT_DIR / f"{base_name}_markers"
    markers_dir.mkdir(parents=True, exist_ok=True)

    h_img, w_img = gray.shape[:2]
    regions = []
    idx = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 0.01 * w_img * h_img:
            continue

        aspect = w / float(h + 1e-6)
        if 0.5 < aspect < 10.0:  # evitar cosas extremadamente alargadas o finas
            crop = image_bgr[y : y + h, x : x + w]
            out_path = markers_dir / f"{base_name}_marker_{idx:02d}.png"
            cv2.imwrite(str(out_path), crop)
            regions.append((x, y, w, h))
            idx += 1

    if regions:
        print(f"[detect_markers] Guardadas {len(regions)} posibles marcas/códigos en: {markers_dir}")
    else:
        print("[detect_markers] No se han encontrado marcas claras.")

    return regions
