import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from config import (
    OUTPUT_DIR,
    PRINTED_MODEL_PATH,
    HANDWRITTEN_MODEL_PATH,
)
from core.preprocessing import preprocess_image
from core.segmentation import segment_characters_from_binary
from core.features import extract_features_for_char_images
from core.classifier import OCRModel
from core.postprocess import rebuild_text_from_segments
from extras.detect_images import detect_and_save_images
from extras.detect_tables import detect_and_export_tables
from extras.detect_markers import detect_and_save_markers


def parse_args():
    parser = argparse.ArgumentParser(description="Mini OCR clásico (sin Tesseract).")
    parser.add_argument("--mode", choices=["printed", "handwritten"], required=True,
                        help="Tipo de texto esperado en la imagen.")
    parser.add_argument("--input", required=True, help="Ruta a la imagen de entrada.")
    parser.add_argument("--out", default=None, help="Ruta del archivo de texto de salida.")
    parser.add_argument("--detect-images", action="store_true",
                        help="Intentar detectar imágenes embebidas y guardarlas aparte.")
    parser.add_argument("--detect-tables", action="store_true",
                        help="Intentar detectar tablas y exportarlas a Markdown.")
    parser.add_argument("--detect-markers", action="store_true",
                        help="Intentar detectar códigos/etiquetas y guardarlos aparte.")
    return parser.parse_args()


def load_model(mode: str) -> OCRModel:
    if mode == "printed":
        model_path = PRINTED_MODEL_PATH
    else:
        model_path = HANDWRITTEN_MODEL_PATH

    if not model_path.exists():
        raise SystemExit(
            f"No se ha encontrado el modelo '{model_path}'.\n"
            f"Primero genera dataset y entrena modelos ejecutando:\n"
            f"  python -m training.generate_synthetic  # para impreso\n"
            f"  python -m training.build_handwritten   # para manuscrito (tras recopilar hojas)\n"
            f"  python -m training.train_models"
        )
    return OCRModel.load(model_path)


def main():
    args = parse_args()
    img_path = Path(args.input)
    if not img_path.exists():
        raise SystemExit(f"Imagen de entrada no encontrada: {img_path}")

    # Cargar imagen en color (para extras) y gris/binaria para OCR
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        raise SystemExit(f"No se pudo leer la imagen: {img_path}")

    binary = preprocess_image(image_bgr)

    # Segmentación en caracteres
    segments = segment_characters_from_binary(binary)

    if not segments:
        print("No se han encontrado caracteres en la imagen.")
        text = ""
    else:
        # Extraer features
        char_images = [seg.char_image for seg in segments]
        X = extract_features_for_char_images(char_images)

        # Cargar modelo y predecir
        model = load_model(args.mode)
        labels = model.predict(X)

        # Reconstruir texto
        text = rebuild_text_from_segments(segments, labels)

    # Guardar salida texto
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = OUTPUT_DIR / (img_path.stem + "_ocr.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Texto OCR guardado en: {out_path}")

    # Extras
    if args.detect_images:
        detect_and_save_images(image_bgr, img_path.stem)

    if args.detect_tables:
        detect_and_export_tables(image_bgr, img_path.stem)

    if args.detect_markers:
        detect_and_save_markers(image_bgr, img_path.stem)


if __name__ == "__main__":
    main()
