from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from config import DATA_DIR


def draw_grid_table(
        rows: int,
        cols: int,
        cell_w: int = 120,
        cell_h: int = 80,
        margin: int = 40,
        line_thickness: int = 2,
) -> tuple[cv2.Mat, tuple[int, int, int, int]]:
    """
    Crea una imagen en blanco con una rejilla negra de (rows x cols).
    Devuelve:
      - imagen BGR
      - bounding box de la tabla (x0, y0, x1, y1)
    """
    width = cols * cell_w + 2 * margin
    height = rows * cell_h + 2 * margin

    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # blanco

    x0 = margin
    y0 = margin
    x1 = margin + cols * cell_w
    y1 = margin + rows * cell_h

    # Líneas verticales
    for c in range(cols + 1):
        x = x0 + c * cell_w
        cv2.line(img, (x, y0), (x, y1), (0, 0, 0), thickness=line_thickness)

    # Líneas horizontales
    for r in range(rows + 1):
        y = y0 + r * cell_h
        cv2.line(img, (x0, y), (x1, y), (0, 0, 0), thickness=line_thickness)

    return img, (x0, y0, x1, y1)


def put_centered_text(
        img: cv2.Mat,
        text: str,
        cell_rect: tuple[int, int, int, int],
        font_scale: float = 1.0,
        thickness: int = 2,
):
    """
    Escribe texto centrado en una celda (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = cell_rect
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    org = (cx - tw // 2, cy + th // 2)
    cv2.putText(img, text, org, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def generate_multiplication_table_5(out_path: Path):
    """
    Genera la tabla del 5 limpia y perfectamente OCR-friendly.
    Formato:
      5 | x | 1 | 5
      5 | x | 2 | 10
      ...
    """
    rows, cols = 10, 4
    img, (x0, y0, x1, y1) = draw_grid_table(rows, cols)

    cell_w = (x1 - x0) // cols
    cell_h = (y1 - y0) // rows

    for i in range(1, rows + 1):
        row = i - 1

        # 5
        put_centered_text(img, "5",
                          (x0 + 0 * cell_w, y0 + row * cell_h,
                           x0 + 1 * cell_w, y0 + (row + 1) * cell_h),
                          font_scale=1.2, thickness=2)

        # x
        put_centered_text(img, "x",
                          (x0 + 1 * cell_w, y0 + row * cell_h,
                           x0 + 2 * cell_w, y0 + (row + 1) * cell_h),
                          font_scale=1.2, thickness=2)

        # multiplicador
        put_centered_text(img, str(i),
                          (x0 + 2 * cell_w, y0 + row * cell_h,
                           x0 + 3 * cell_w, y0 + (row + 1) * cell_h),
                          font_scale=1.2, thickness=2)

        # resultado
        put_centered_text(img, str(5 * i),
                          (x0 + 3 * cell_w, y0 + row * cell_h,
                           x0 + 4 * cell_w, y0 + (row + 1) * cell_h),
                          font_scale=1.2, thickness=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[generate_table_samples] Tabla del 5 guardada en: {out_path}")


def generate_empty_5x5_table(out_path: Path):
    """
    Genera una tabla vacía 5×5 (solo rejilla).
    Muy útil para mostrar detección robusta de tablas.
    """
    rows, cols = 5, 5
    img, _ = draw_grid_table(rows, cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[generate_table_samples] Tabla vacía 5x5 guardada en: {out_path}")


def main():
    samples_dir = DATA_DIR / "samples"

    mult5_path = samples_dir / "tabla_multiplicar_5.png"
    empty_path = samples_dir / "tabla_vacia_5x5.png"

    generate_multiplication_table_5(mult5_path)
    generate_empty_5x5_table(empty_path)

    print("[generate_table_samples] Tablas de ejemplo generadas correctamente.")


if __name__ == "__main__":
    main()
