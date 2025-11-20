from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from config import (
    DATA_DIR,
    HANDWRITTEN_CHARS,
    CHAR_SIZE,
)
from core.preprocessing import preprocess_image, normalize_char_image
from core.features import extract_features_for_char_images


def _split_sheet_into_cells(sheet_gray: np.ndarray, n_rows: int, n_cols: int) -> List[np.ndarray]:
    h, w = sheet_gray.shape[:2]
    cell_h = h // n_rows
    cell_w = w // n_cols

    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * cell_h
            x0 = c * cell_w
            cell = sheet_gray[y0 : y0 + cell_h, x0 : x0 + cell_w]
            cells.append(cell)
    return cells


def _extract_chars_from_cell(cell_gray: np.ndarray) -> List[np.ndarray]:
    # Preprocesar como si fuera imagen completa
    cell_bgr = cv2.cvtColor(cell_gray, cv2.COLOR_GRAY2BGR)
    binary = preprocess_image(cell_bgr)

    # Componentes conectados dentro de la celda
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    chars = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 30:
            continue
        crop = binary[y : y + h, x : x + w]
        chars.append(crop)
    return chars


def build_handwritten_dataset_from_sheets(
    sheets_dir: Path, out_path: Path, n_rows: int = 8, n_cols: int = 8
) -> None:
    """Construye un dataset manuscrito a partir de hojas de plantilla.

    La suposición es:
      - Cada fila/columna de la plantilla corresponde a un carácter conocido.
      - HANDWRITTEN_CHARS se rellena por filas y columnas.
    """
    chars = HANDWRITTEN_CHARS
    if len(chars) > n_rows * n_cols:
        raise SystemExit("HANDWRITTEN_CHARS tiene más caracteres que celdas definidas")

    labels = []
    images = []

    sheet_paths = sorted(list(sheets_dir.glob("*.png")) + list(sheets_dir.glob("*.jpg")) + list(sheets_dir.glob("*.jpeg")))
    if not sheet_paths:
        raise SystemExit(f"No se han encontrado hojas manuscritas en {sheets_dir}")

    for sheet_path in sheet_paths:
        print(f"[build_handwritten] Procesando hoja: {sheet_path.name}")
        img_bgr = cv2.imread(str(sheet_path))
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        cells = _split_sheet_into_cells(gray, n_rows=n_rows, n_cols=n_cols)
        if len(cells) < len(chars):
            raise SystemExit("La hoja no tiene suficientes celdas para todos los caracteres de HANDWRITTEN_CHARS")

        for idx, ch in enumerate(chars):
            cell = cells[idx]
            cell_chars = _extract_chars_from_cell(cell)
            for cc in cell_chars:
                norm = normalize_char_image(cc)
                images.append(norm)
                labels.append(ch)

    if not images:
        raise SystemExit("No se han encontrado caracteres manuscritos en las hojas.")

    X = extract_features_for_char_images(images)
    y = np.array(labels)
    np.savez_compressed(out_path, X=X, y=y)

    print(f"[build_handwritten] Dataset manuscrito guardado en: {out_path}  (X.shape={X.shape})")


def main():
    parser = argparse.ArgumentParser(description="Construir dataset manuscrito a partir de hojas de plantilla.")
    parser.add_argument(
        "--sheets-dir",
        type=str,
        default=str(DATA_DIR / "raw_handwritten"),
        help="Directorio con hojas manuscritas escaneadas.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DATA_DIR / "processed" / "handwritten_dataset.npz"),
        help="Ruta del .npz resultante.",
    )
    parser.add_argument("--rows", type=int, default=8, help="Número de filas en la plantilla.")
    parser.add_argument("--cols", type=int, default=8, help="Número de columnas en la plantilla.")
    args = parser.parse_args()

    sheets_dir = Path(args.sheets_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_handwritten_dataset_from_sheets(
        sheets_dir=sheets_dir,
        out_path=out_path,
        n_rows=args.rows,
        n_cols=args.cols,
    )


if __name__ == "__main__":
    main()
