from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from config import (
    DATA_DIR,
    HANDWRITTEN_CHARS,
    CHAR_SIZE,
)
from core.preprocessing import preprocess_image, normalize_char_image
from core.features import extract_features_for_char_images


def render_handwritten_char(char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """
    Genera una imagen simulada de un carácter manuscrito mediante:
    - rotación aleatoria
    - posición aleatoria
    - grosor/blur
    - ruido
    """
    img = Image.new("L", (120, 120), color=255)
    draw = ImageDraw.Draw(img)

    # Random baseline shifts (simular escritura humana)
    x = random.randint(10, 25)
    y = random.randint(10, 25)

    draw.text((x, y), char, font=font, fill=0)

    # Rotación leve
    angle = random.uniform(-12, 12)
    img = img.rotate(angle, expand=True, fillcolor=255)

    # Blur ligero
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.1)))

    # Ruido
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)

    # Convertir a BGR para procesado
    img_bgr = np.stack([np.array(img)] * 3, axis=-1)

    # Binarizar + normalizar como en OCR real
    binary = preprocess_image(img_bgr)
    norm = normalize_char_image(binary)

    return norm


def generate_synthetic_handwritten_dataset(
        font_path: Path,
        out_path: Path,
        samples_per_char: int = 80,
):
    """
    Crea un dataset manuscrito sintético usando una fuente manuscrita.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype(str(font_path), size=80)
    except Exception as e:
        raise SystemExit(f"No se pudo cargar la fuente manuscrita: {font_path}\nError: {e}")

    images = []
    labels = []

    random.seed(42)
    np.random.seed(42)

    print(f"[generate_synthetic_handwritten] Fuente: {font_path.name}")
    print(f"[generate_synthetic_handwritten] Caracteres: {HANDWRITTEN_CHARS}")

    for char in HANDWRITTEN_CHARS:
        print(f" → Generando '{char}' ...")
        for _ in range(samples_per_char):
            img_norm = render_handwritten_char(char, font)
            images.append(img_norm)
            labels.append(char)

    X = extract_features_for_char_images(images)
    y = np.array(labels)

    np.savez_compressed(out_path, X=X, y=y)
    print(f"[generate_synthetic_handwritten] Dataset generado: {out_path}  (X.shape={X.shape})")


def main():
    parser = argparse.ArgumentParser(description="Dataset manuscrito sintético")
    parser.add_argument("--font", type=str, required=True, help="Ruta a la fuente manuscrita")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DATA_DIR / "processed" / "handwritten_synthetic.npz"),
    )
    parser.add_argument("--samples-per-char", type=int, default=80)
    args = parser.parse_args()

    font_path = Path(args.font)
    out_path = Path(args.out)

    generate_synthetic_handwritten_dataset(
        font_path=font_path,
        out_path=out_path,
        samples_per_char=args.samples_per_char,
    )


if __name__ == "__main__":
    main()
