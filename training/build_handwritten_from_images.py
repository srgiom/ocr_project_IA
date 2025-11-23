from __future__ import annotations

import argparse
from pathlib import Path
import re
import os

import numpy as np
import cv2

from config import DATA_DIR
from core.preprocessing import preprocess_image, normalize_char_image
from core.features import extract_features_for_char_images


def extract_label_from_filename(fname: str) -> str:
    """
    Extrae el carácter a partir del nombre del archivo.
    Ejemplos:
      - 'A_sergio.png' -> 'A'
      - 'b_23.jpg'   -> 'B' (lo pasa a mayúscula)
      - '7_algo.png' -> '7'
    """
    match = re.match(r"([A-Za-z0-9])", fname)
    if not match:
        raise ValueError(f"No se pudo extraer letra de: {fname}")
    return match.group(1).upper()


def build_dataset_from_images(images_dir: Path, out_path: Path):
    """
    Crea un dataset manuscrito desde un directorio donde cada archivo
    es un carácter manuscrito ya recortado.

    Recorre recursivamente todas las subcarpetas.
    """
    print(f"[build_from_images] Leyendo dataset desde: {images_dir}")

    images = []
    labels = []
    count = 0

    for root, _, files in os.walk(images_dir):
        root_path = Path(root)
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            img_path = root_path / fname

            try:
                label = extract_label_from_filename(fname)
            except Exception:
                print(f"[WARN] Ignorando archivo no válido: {img_path.name}")
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[WARN] No se pudo leer: {img_path}")
                continue

            # Preprocesado igual que en el OCR real
            binary = preprocess_image(img_bgr)
            norm = normalize_char_image(binary)

            images.append(norm)
            labels.append(label)
            count += 1

    if not images:
        raise SystemExit("[build_from_images] No se encontraron imágenes válidas.")

    print(f"[build_from_images] Total caracteres cargados: {count}")

    X = extract_features_for_char_images(images)
    y = np.array(labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)

    print(f"[build_from_images] Dataset guardado en: {out_path}")
    print(f"[build_from_images] X.shape = {X.shape}")


def main():
    parser = argparse.ArgumentParser(description="Construir dataset manuscrito desde imágenes sueltas.")
    parser.add_argument("--dir", type=str, required=True, help="Directorio con imágenes tipo A_nombre.png")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DATA_DIR / "processed" / "handwritten_from_images.npz"),
    )
    args = parser.parse_args()

    build_dataset_from_images(
        images_dir=Path(args.dir),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
