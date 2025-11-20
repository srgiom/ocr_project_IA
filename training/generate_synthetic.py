from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    DATA_DIR,
    PRINTED_CHARS,
    CHAR_SIZE,
)
from core.preprocessing import preprocess_image, normalize_char_image
from core.features import extract_features_for_char_images


def _list_fonts(fonts_dir: Path):
    fonts = []
    for ext in ("*.ttf", "*.otf"):
        fonts.extend(fonts_dir.glob(ext))
    return fonts


def generate_synthetic_dataset(fonts_dir: Path, out_path: Path, samples_per_char: int = 50) -> None:
    random.seed(42)
    np.random.seed(42)

    fonts = _list_fonts(fonts_dir)
    if not fonts:
        raise SystemExit(f"No se han encontrado fuentes en {fonts_dir} (ttf/otf).")

    char_images = []
    labels = []

    for font_path in fonts:
        print(f"[generate_synthetic] Usando fuente: {font_path.name}")
        # Probamos varios tamaños por fuente
        for ch in PRINTED_CHARS:
            for _ in range(samples_per_char):
                img = Image.new("L", (160, 160), color=255)
                draw = ImageDraw.Draw(img)

                font_size = random.randint(60, 120)
                try:
                    font = ImageFont.truetype(str(font_path), font_size)
                except Exception:
                    continue

                bbox = draw.textbbox((0, 0), ch, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                x = (img.width - w) // 2
                y = (img.height - h) // 2
                draw.text((x, y), ch, font=font, fill=0)

                # Ligera rotación / ruido
                angle = random.uniform(-5, 5)
                img = img.rotate(angle, expand=True, fillcolor=255)

                img_np = np.array(img, dtype=np.uint8)
                # Convertir a BGR para reutilizar preprocess_image
                img_bgr = np.stack([img_np] * 3, axis=-1)
                binary = preprocess_image(img_bgr)

                # Normalizar y añadir a la lista
                norm = normalize_char_image(binary)
                char_images.append(norm)
                labels.append(ch)

    if not char_images:
        raise SystemExit("No se han generado imágenes sintéticas. Revisa las fuentes.")

    X = extract_features_for_char_images(char_images)
    y = np.array(labels)

    np.savez_compressed(out_path, X=X, y=y)
    print(f"[generate_synthetic] Dataset guardado en: {out_path}  (X.shape={X.shape})")


def main():
    parser = argparse.ArgumentParser(description="Generador de dataset sintético de caracteres impresos.")
    parser.add_argument("--fonts-dir", type=str, default=str(DATA_DIR / "fonts"), help="Directorio con fuentes ttf/otf.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DATA_DIR / "processed" / "printed_synthetic.npz"),
        help="Ruta del .npz de salida.",
    )
    parser.add_argument("--samples-per-char", type=int, default=50, help="Muestras sintéticas por carácter y fuente.")
    args = parser.parse_args()

    fonts_dir = Path(args.fonts_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generate_synthetic_dataset(fonts_dir, out_path, samples_per_char=args.samples_per_char)


if __name__ == "__main__":
    main()
