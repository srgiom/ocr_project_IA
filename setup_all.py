from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from typing import Tuple


def find_fonts(fonts_dir: Path) -> Tuple[Path, Path]:
    """
    Busca fuentes en data/fonts y devuelve:
    (printed_font, handwritten_font)
    """
    if not fonts_dir.exists():
        raise SystemExit(f"[setup_all] No existe el directorio de fuentes: {fonts_dir}")

    fonts = []
    for ext in ("*.ttf", "*.otf", "*.ttc"):
        fonts.extend(fonts_dir.glob(ext))

    if not fonts:
        raise SystemExit(
            f"[setup_all] No se han encontrado fuentes en {fonts_dir}.\n"
            "Necesitas al menos 1 impresa y 1 manuscrita."
        )

    printed_font = fonts[0]
    handwritten_font = fonts[-1]

    # Elegir manuscrita por nombre si es posible
    for f in fonts:
        n = f.name.lower()
        if any(key in n for key in ["hand", "script", "caveat", "brush"]):
            handwritten_font = f
            break

    print(f"[setup_all] Fuente IMPRESA:     {printed_font.name}")
    print(f"[setup_all] Fuente MANUSCRITA:  {handwritten_font.name}")

    return printed_font, handwritten_font


def generate_datasets(project_root: Path, printed_font: Path, handwritten_font: Path):
    """
    Genera datasets:
    - printed_synthetic.npz
    - handwritten_synthetic.npz
    - handwritten_from_images.npz (si existe my_dataset)
    - handwritten_merged.npz (fusión sintético + real si ambos existen)
    """
    sys.path.insert(0, str(project_root))

    from config import DATA_DIR
    from training.generate_synthetic import generate_synthetic_dataset
    from training.generate_synthetic_handwritten import generate_synthetic_handwritten_dataset
    from training.build_handwritten_from_images import build_dataset_from_images

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    printed_npz = processed_dir / "printed_synthetic.npz"
    handwritten_synth_npz = processed_dir / "handwritten_synthetic.npz"
    handwritten_real_npz = processed_dir / "handwritten_from_images.npz"
    handwritten_merged_npz = processed_dir / "handwritten_merged.npz"

    # ----- Dataset impreso -----
    print("\n[setup_all] === Dataset IMPRESO sintético ===")
    generate_synthetic_dataset(
        fonts_dir=printed_font.parent,
        out_path=printed_npz,
        samples_per_char=50,
    )

    # ----- Dataset manuscrito sintético -----
    print("\n[setup_all] === Dataset MANUSCRITO sintético ===")
    generate_synthetic_handwritten_dataset(
        font_path=handwritten_font,
        out_path=handwritten_synth_npz,
        samples_per_char=80,
    )

    # ----- Dataset manuscrito real desde imágenes -----
    raw_real_dir = DATA_DIR / "raw_handwritten" / "my_dataset"
    handwritten_real_exists = False

    if raw_real_dir.exists():
        print("\n[setup_all] === Dataset MANUSCRITO REAL detectado ===")
        print(f"Carpeta encontrada: {raw_real_dir}")

        build_dataset_from_images(
            images_dir=raw_real_dir,
            out_path=handwritten_real_npz
        )
        handwritten_real_exists = True
    else:
        print("\n[setup_all] No hay dataset manuscrito real (raw_handwritten/my_dataset). Solo sintético.")

    # ----- Fusionar sintético + real si existe -----
    if handwritten_real_exists:
        import numpy as np
        print("\n[setup_all] === Fusionando dataset manuscrito sintético + real ===")

        synth = np.load(handwritten_synth_npz)
        real = np.load(handwritten_real_npz)

        X = np.concatenate([synth["X"], real["X"]], axis=0)
        y = np.concatenate([synth["y"], real["y"]], axis=0)

        np.savez_compressed(handwritten_merged_npz, X=X, y=y)

        print(f"[setup_all] Dataset fusionado guardado en: {handwritten_merged_npz}")

        return printed_npz, handwritten_merged_npz

    return printed_npz, handwritten_synth_npz


def train_models(project_root: Path, printed_npz: Path, handwritten_npz: Path):
    sys.path.insert(0, str(project_root))

    from config import PRINTED_MODEL_PATH, HANDWRITTEN_MODEL_PATH
    from training.train_models import train_and_save_models

    print("\n[setup_all] === Entrenando modelos ===")
    train_and_save_models(
        printed_npz=printed_npz,
        handwritten_npz=handwritten_npz,
        printed_out=PRINTED_MODEL_PATH,
        handwritten_out=HANDWRITTEN_MODEL_PATH,
        model_type="knn",
    )


def generate_test_images(project_root: Path, printed_font: Path, handwritten_font: Path):
    """Genera imágenes de test para impreso y manuscrito."""
    sys.path.insert(0, str(project_root))

    from config import DATA_DIR
    from generators.generate_test_image import render_text_image
    from generators.generate_handwritten_test_image import (
        generate_handwritten_test_image,
    )

    samples_dir = DATA_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    printed_img = samples_dir / "test_ocr_setup.png"
    handwritten_img = samples_dir / "test_handwritten_setup.png"

    print("\n[setup_all] === Generando imagen de prueba IMPRESA ===")
    render_text_image(
        text="HOLA ESTO ES UNA PRUEBA DE OCR IMPRESO\nEXAMEN IA",
        font_path=printed_font,
        out_path=printed_img,
    )

    print("\n[setup_all] === Generando imagen de prueba MANUSCRITA ===")
    generate_handwritten_test_image(
        text="Hola esto es una prueba de OCR manuscrito\nexamen IA",
        font_path=handwritten_font,
        out_path=handwritten_img,
    )

    return printed_img, handwritten_img


def run_ocr(project_root: Path, printed_img: Path, handwritten_img: Path):
    from config import OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    printed_out = OUTPUT_DIR / "setup_printed_test.txt"
    handwritten_out = OUTPUT_DIR / "setup_handwritten_test.txt"

    print("\n[setup_all] === OCR IMPRESO ===")
    subprocess.run([
        sys.executable, "main.py",
        "--mode", "printed",
        "--input", str(printed_img),
        "--out", str(printed_out),
    ], cwd=str(project_root), check=True)

    print("\n[setup_all] === OCR MANUSCRITO ===")
    subprocess.run([
        sys.executable, "main.py",
        "--mode", "handwritten",
        "--input", str(handwritten_img),
        "--out", str(handwritten_out),
    ], cwd=str(project_root), check=True)

    print(f"\n[setup_all] Resultados listos en:")
    print(f" - {printed_out}")
    print(f" - {handwritten_out}")


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    print("[setup_all] Proyecto raíz:", project_root)

    fonts_dir = project_root / "data" / "fonts"
    printed_font, handwritten_font = find_fonts(fonts_dir)

    printed_npz, handwritten_npz = generate_datasets(project_root, printed_font, handwritten_font)

    train_models(project_root, printed_npz, handwritten_npz)

    printed_img, handwritten_img = generate_test_images(project_root, printed_font, handwritten_font)

    run_ocr(project_root, printed_img, handwritten_img)

    print("\n[setup_all] === COMPLETADO ===")
    print("Puedes ahora usar main.py con tus imágenes.")


if __name__ == "__main__":
    main()
