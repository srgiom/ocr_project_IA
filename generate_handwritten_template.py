from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import HANDWRITTEN_CHARS


def generate_template(
        out_path: Path,
        rows: int = 8,
        cols: int = 8,
        img_width: int = 2480,   # A4 a 300dpi aprox (ancho)
        img_height: int = 3508,  # A4 a 300dpi aprox (alto)
        margin: int = 150,
        label_font_size: int = 80,
        hint_font_size: int = 40,
) -> None:
    """
    Genera una hoja plantilla con cuadrícula rows x cols.
    Cada celda tiene asignado un carácter de HANDWRITTEN_CHARS en orden fila-columna.

    - Imprime esta hoja.
    - En cada celda, los compañeros escriben varias veces ese carácter.
    - Luego escaneas la hoja y la usas con build_handwritten.py.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Lienzo en blanco (blanco puro)
    img = Image.new("L", (img_width, img_height), color=255)
    draw = ImageDraw.Draw(img)

    # Intentar usar una fuente cualquiera del sistema (si falla, usa la por defecto)
    try:
        # Puedes cambiar esta ruta a una de tus fuentes si quieres algo más bonito
        font_label = ImageFont.truetype("Arial.ttf", label_font_size)
        font_hint = ImageFont.truetype("Arial.ttf", hint_font_size)
    except Exception:
        font_label = ImageFont.load_default()
        font_hint = ImageFont.load_default()

    # Zona útil (dejamos márgenes)
    x0 = margin
    y0 = margin
    x1 = img_width - margin
    y1 = img_height - margin

    cell_w = (x1 - x0) / cols
    cell_h = (y1 - y0) / rows

    # Dibujar rejilla
    for c in range(cols + 1):
        x = int(x0 + c * cell_w)
        draw.line([(x, y0), (x, y1)], fill=0, width=2)

    for r in range(rows + 1):
        y = int(y0 + r * cell_h)
        draw.line([(x0, y), (x1, y)], fill=0, width=2)

    # Poner título arriba
    title = "PLANTILLA OCR MANUSCRITO - ESCRIBE AQUI LOS CARACTERES"
    try:
        bbox_title = draw.textbbox((0, 0), title, font=font_hint)
        tw = bbox_title[2] - bbox_title[0]
        th = bbox_title[3] - bbox_title[1]
    except Exception:
        tw, th = draw.textsize(title, font=font_hint)

    draw.text(
        ((img_width - tw) // 2, margin // 3),
        title,
        font=font_hint,
        fill=0,
    )

    # Poner letra objetivo en cada celda
    chars = HANDWRITTEN_CHARS
    max_cells = rows * cols
    n_chars = min(len(chars), max_cells)

    for idx in range(n_chars):
        ch = chars[idx]
        row = idx // cols
        col = idx % cols

        cell_x0 = x0 + col * cell_w
        cell_y0 = y0 + row * cell_h
        cell_x1 = cell_x0 + cell_w
        cell_y1 = cell_y0 + cell_h

        # Posición del carácter (arriba centrado de la celda)
        try:
            bbox = draw.textbbox((0, 0), ch, font=font_label)
            cw = bbox[2] - bbox[0]
            ch_h = bbox[3] - bbox[1]
        except Exception:
            cw, ch_h = draw.textsize(ch, font=font_label)

        tx = int(cell_x0 + (cell_w - cw) / 2)
        ty = int(cell_y0 + 10)  # un poco por debajo del borde superior

        draw.text((tx, ty), ch, font=font_label, fill=0)

        # Opcional: texto guía pequeño debajo
        hint = f"Escribe varias veces '{ch}'"
        try:
            bbox_hint = draw.textbbox((0, 0), hint, font=font_hint)
            hw = bbox_hint[2] - bbox_hint[0]
            hh = bbox_hint[3] - bbox_hint[1]
        except Exception:
            hw, hh = draw.textsize(hint, font=font_hint)

        hx = int(cell_x0 + (cell_w - hw) / 2)
        hy = ty + ch_h + 5
        draw.text((hx, hy), hint, font=font_hint, fill=0)

    img.save(str(out_path))
    print(f"[generate_handwritten_template] Plantilla guardada en: {out_path}")
    print("Imprime esta hoja y pídele a la gente que rellene las celdas con los caracteres indicados.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generar hoja plantilla para capturar texto manuscrito.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw_handwritten/plantilla_handwritten.png",
        help="Ruta de la imagen de plantilla de salida.",
    )
    parser.add_argument("--rows", type=int, default=8, help="Número de filas de la cuadrícula.")
    parser.add_argument("--cols", type=int, default=8, help="Número de columnas de la cuadrícula.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    generate_template(
        out_path=out_path,
        rows=args.rows,
        cols=args.cols,
    )


if __name__ == "__main__":
    main()
