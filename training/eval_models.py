from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import DATA_DIR, PRINTED_CHARS, HANDWRITTEN_CHARS


def _load_npz(path: Path):
    if not path.exists():
        raise SystemExit(f"No se ha encontrado el dataset: {path}")
    data = np.load(path)
    return data["X"], data["y"]


def eval_dataset(npz_path: Path, charset: str):
    X, y_chars = _load_npz(npz_path)
    # Esto asume que se usará solamente para experimentar (no hay modelo cargado aquí).
    # De forma práctica, esta plantilla se puede adaptar para comparar verdad/previsto.
    print(f"Dataset: {npz_path}")
    print(f"  n_samples: {X.shape[0]}, n_features: {X.shape[1]}")
    uniq, counts = np.unique(y_chars, return_counts=True)
    print("Distribución de clases:")
    for ch, c in zip(uniq, counts):
        print(f"  '{ch}': {c}")


def main():
    parser = argparse.ArgumentParser(description="Pequeña utilidad para inspeccionar datasets de OCR.")
    parser.add_argument(
        "--printed-dataset",
        type=str,
        default=str(DATA_DIR / "processed" / "printed_synthetic.npz"),
    )
    parser.add_argument(
        "--handwritten-dataset",
        type=str,
        default=str(DATA_DIR / "processed" / "handwritten_dataset.npz"),
    )
    args = parser.parse_args()

    print("=== Dataset impreso ===")
    eval_dataset(Path(args.printed_dataset), PRINTED_CHARS)

    print("\n=== Dataset manuscrito ===")
    eval_dataset(Path(args.handwritten_dataset), HANDWRITTEN_CHARS)


if __name__ == "__main__":
    main()
