from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import (
    DATA_DIR,
    MODELS_DIR,
    PRINTED_CHARS,
    HANDWRITTEN_CHARS,
    PRINTED_MODEL_PATH,
    HANDWRITTEN_MODEL_PATH,
)
from core.classifier import OCRModel, train_knn, train_linear_svc


def _load_npz(path: Path):
    if not path.exists():
        raise SystemExit(f"No se ha encontrado el dataset: {path}")
    data = np.load(path)
    return data["X"], data["y"]


def train_and_save_models(
        printed_npz: Path,
        handwritten_npz: Path,
        printed_out: Path,
        handwritten_out: Path,
        model_type: str = "knn",
):
    # =========================
    #  MODELO IMPRESO (OBLIG.)
    # =========================
    Xp, yp_chars = _load_npz(printed_npz)
    char_to_idx_p = {ch: i for i, ch in enumerate(PRINTED_CHARS)}
    yp = np.array([char_to_idx_p[ch] for ch in yp_chars], dtype=int)

    if model_type == "knn":
        model_p = train_knn(Xp, yp, n_neighbors=5)
    else:
        model_p = train_linear_svc(Xp, yp)

    ocrm_p = OCRModel(model=model_p, charset=PRINTED_CHARS)
    ocrm_p.save(printed_out)
    print(f"[train_models] Modelo impreso guardado en: {printed_out}")

    # ==============================
    #  MODELO MANUSCRITO (OPCIONAL)
    # ==============================
    if not handwritten_npz.exists():
        print(
            f"[train_models] AVISO: no se ha encontrado el dataset manuscrito "
            f"({handwritten_npz}). Se omite el entrenamiento manuscrito."
        )
        return

    Xh, yh_chars = _load_npz(handwritten_npz)
    char_to_idx_h = {ch: i for i, ch in enumerate(HANDWRITTEN_CHARS)}
    yh = np.array([char_to_idx_h[ch] for ch in yh_chars], dtype=int)

    if model_type == "knn":
        model_h = train_knn(Xh, yh, n_neighbors=5)
    else:
        model_h = train_linear_svc(Xh, yh)

    ocrm_h = OCRModel(model=model_h, charset=HANDWRITTEN_CHARS)
    ocrm_h.save(handwritten_out)
    print(f"[train_models] Modelo manuscrito guardado en: {handwritten_out}")


def main():
    parser = argparse.ArgumentParser(description="Entrena modelos OCR (impreso + manuscrito).")
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
    parser.add_argument(
        "--model-type",
        choices=["knn", "svc"],
        default="knn",
        help="Tipo de clasificador: KNN o LinearSVC.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_and_save_models(
        printed_npz=Path(args.printed_dataset),
        handwritten_npz=Path(args.handwritten_dataset),
        printed_out=PRINTED_MODEL_PATH,
        handwritten_out=HANDWRITTEN_MODEL_PATH,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
