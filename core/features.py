from __future__ import annotations

import numpy as np
import cv2

from config import (
    CHAR_SIZE,
    HOG_N_BINS,
    HOG_CELL_SIZE,
    HOG_BLOCK_SIZE,
)


def _compute_gradients(image: np.ndarray):
    # Sobel gradientes
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return mag, ang


def _hog_descriptor(image: np.ndarray) -> np.ndarray:
    """Implementación sencilla de HOG para un parche 2D (binario o gris)."""
    # Asegurar tamaño esperado
    if image.shape != CHAR_SIZE:
        image = cv2.resize(image, CHAR_SIZE, interpolation=cv2.INTER_AREA)

    mag, ang = _compute_gradients(image)

    cell_h, cell_w = HOG_CELL_SIZE
    n_cells_y = image.shape[0] // cell_h
    n_cells_x = image.shape[1] // cell_w

    # Cuantizar ángulos en bins
    bin_width = 180.0 / HOG_N_BINS  # usaremos [0,180)
    bins = np.int32(ang // bin_width)
    bins[bins >= HOG_N_BINS] = HOG_N_BINS - 1

    # Histograma por celda
    hist = np.zeros((n_cells_y, n_cells_x, HOG_N_BINS), dtype=np.float32)

    for y in range(n_cells_y):
        for x in range(n_cells_x):
            cell_mag = mag[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]
            cell_bins = bins[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]
            for i in range(cell_mag.shape[0]):
                for j in range(cell_mag.shape[1]):
                    b = cell_bins[i, j]
                    hist[y, x, b] += cell_mag[i, j]

    # Normalización por bloques
    by, bx = HOG_BLOCK_SIZE
    eps = 1e-6
    block_features = []
    for y in range(n_cells_y - by + 1):
        for x in range(n_cells_x - bx + 1):
            block = hist[y:y+by, x:x+bx, :].ravel()
            norm = np.sqrt(np.sum(block ** 2) + eps**2)
            block_features.append(block / norm)

    return np.concatenate(block_features, axis=0)


def extract_features_for_char_images(char_images):
    """Devuelve una matriz (n_samples, n_features) para una lista de parches de caracteres."""
    feats = [ _hog_descriptor(img) for img in char_images ]
    return np.vstack(feats)
