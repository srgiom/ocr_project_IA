"""
python generate_test_image.py \
  --text "HOLA MUNDO ESTO ES UNA PRUEBA DE OCR IMPRESO" \
  --font data/fonts/FreeSans.ttf \
  --out data/samples/test_ocr_1.png \
  --width 1200 \
  --height 400 \
  --font-size 72
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_text_image(
        text: str,
        font_path: Path,
        out_path: Path,
        img_width: int = 1200,
        img_height: int = 400,
        font_size: int = 72,
        margin: int = 40,
        line_spacing: float = 1.2,
        bg_color: int = 255,
        fg_color: int = 0,
) -> None:
    """
    Genera una imagen con texto impreso (para pruebas del OCR).

    - text: texto completo (puede contener '\n')
    - font_path: ruta a .ttf/.otf
    - out_path: imagen de salida (.png/.jpg)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Crear lienzo en blanco
    img = Image.new("L", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception as e:
        raise SystemExit(f"No se pudo cargar la fuente '{font_path}': {e}")

    # Soportar saltos de línea manuales
    lines = text.split("\n")

    # Calcular altura de línea usando textbbox (Pillow moderno)
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    base_line_height = bbox[3] - bbox[1]
    line_height = int(base_line_height * line_spacing)

    y = margin
    for line in lines:
        # Wrap manual muy simple si la línea es demasiado larga: cortamos por palabras
        words = line.split(" ")
        current = ""
        for w in words:
            candidate = (current + " " + w).strip()
            bbox_c = draw.textbbox((0, 0), candidate, font=font)
            width_c = bbox_c[2] - bbox_c[0]

            if margin + width_c > img_width - margin and current != "":
                # Pintar la línea actual
                draw.text((margin, y), current, font=font, fill=fg_color)
                y += line_height
                current = w
            else:
                current = candidate

        # Pintar lo que quede
        if current:
            draw.text((margin, y), current, font=font, fill=fg_color)
            y += line_height

        # Línea en blanco entre párrafos
        y += int(line_height * 0.1)

        if y + line_height > img_height - margin:
            break  # no cabe más texto en la imagen

    img.save(str(out_path))
    print(f"[generate_test_image] Imagen de prueba guardada en: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generar imagen de texto para pruebas de OCR.")
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Texto a renderizar (si no se usa --text-file).",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        required=False,
        help="Archivo .txt con el texto a renderizar.",
    )
    parser.add_argument(
        "--font",
        type=str,
        required=True,
        help="Ruta a la fuente .ttf/.otf/.ttc (por ejemplo data/fonts/FreeSans.ttf).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/samples/test_image.png",
        help="Ruta de la imagen de salida.",
    )
    parser.add_argument("--width", type=int, default=1200, help="Ancho de la imagen.")
    parser.add_argument("--height", type=int, default=400, help="Alto de la imagen.")
    parser.add_argument("--font-size", type=int, default=72, help="Tamaño de fuente.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.text and not args.text_file:
        raise SystemExit("Debes indicar --text o --text-file")

    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            raise SystemExit(f"No se encontró el archivo de texto: {text_path}")
        text = text_path.read_text(encoding="utf-8")
    else:
        text = args.text

    font_path = Path(args.font)
    out_path = Path(args.out)

    render_text_image(
        text=text,
        font_path=font_path,
        out_path=out_path,
        img_width=args.width,
        img_height=args.height,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
