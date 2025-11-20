"""
python generate_handwritten_test_image.py \
  --text "Hola esta es una prueba de OCR manuscrito" \
  --font data/fonts/PatrickHand-Regular.ttf \
  --out data/samples/test_handwritten1.png \
  --font-size 90 \
  --width 1200 \
  --height 450
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def load_handwritten_font(font_dir: Path):
    """Busca fuentes manuscritas dentro de data/fonts y devuelve una al azar."""
    fonts = []
    for ext in ("*.ttf", "*.otf", "*.ttc"):
        fonts.extend(font_dir.glob(ext))
    if not fonts:
        raise SystemExit(f"No se encontraron fuentes manuscritas en {font_dir}")
    return random.choice(fonts)


def simulate_manuscript_style(
        img: Image.Image,
        max_rotation: float = 3.0,
        noise_level: int = 15,
):
    """Aplica efectos para que parezca manuscrito real."""
    # Rotación leve
    angle = random.uniform(-max_rotation, max_rotation)
    img = img.rotate(angle, expand=True, fillcolor=255)

    # Ligero blur
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    # Ruido
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-noise_level, noise_level, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def generate_handwritten_test_image(
        text: str,
        font_path: Path,
        out_path: Path,
        img_width: int = 1200,
        img_height: int = 400,
        font_size: int = 90,
        margin: int = 40,
        line_spacing: float = 1.15,
):
    """Genera una imagen 'handwritten-like' para pruebas de OCR manuscrito."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Blanco
    img = Image.new("L", (img_width, img_height), color=255)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception as e:
        raise SystemExit(f"No se pudo cargar la fuente manuscrita '{font_path}': {e}")

    lines = text.split("\n")

    # Altura de línea
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = int((bbox[3] - bbox[1]) * line_spacing)

    y = margin
    for line in lines:
        # Saltar líneas vacías
        if not line.strip():
            y += line_height
            continue

        # Simular escritura irregular: mover baseline
        baseline_offset = random.randint(-5, 5)

        draw.text(
            (margin, y + baseline_offset),
            line,
            font=font,
            fill=0,
        )

        y += line_height

        if y + line_height > img_height - margin:
            break

    # Aplicar efectos manuscritos
    img = simulate_manuscript_style(img)

    img.save(str(out_path))
    print(f"[generate_handwritten_test_image] Imagen manuscrita generada: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generar imagen simulada manuscrita.")
    parser.add_argument("--text", type=str, required=True, help="Texto manuscrito a simular.")
    parser.add_argument(
        "--font",
        type=str,
        required=True,
        help="Fuente manuscrita (.ttf/.otf) dentro de data/fonts/",
    )
    parser.add_argument("--out", type=str, default="data/samples/handwritten_test.png")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--font-size", type=int, default=90)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_handwritten_test_image(
        text=args.text,
        font_path=Path(args.font),
        out_path=Path(args.out),
        img_width=args.width,
        img_height=args.height,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
