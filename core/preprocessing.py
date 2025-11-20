from __future__ import annotations

import cv2
import numpy as np

from config import CHAR_SIZE


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR en una binaria 'limpia' para segmentación de texto.

    Devuelve una imagen binaria uint8 con valores 0 (fondo) y 255 (texto).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Suavizado ligero para reducir ruido
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarización adaptativa suele ir bien con manuscrito
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=10,
    )

    # Pequeña apertura para eliminar puntitos aislados
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


def normalize_char_image(char_img: np.ndarray, size: tuple[int, int] = CHAR_SIZE) -> np.ndarray:
    """Normaliza una imagen de carácter a tamaño fijo con padding y centrado aproximado.

    Se asume char_img como binaria uint8 (0 fondo, 255 trazo).
    """
    # Recortar bounding box ajustado
    ys, xs = np.where(char_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        # Imagen vacía: devolver parche blanco
        return np.zeros(size, dtype=np.uint8)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = char_img[y_min : y_max + 1, x_min : x_max + 1]

    # Mantener aspecto, redimensionando al tamaño objetivo con padding
    target_h, target_w = size
    h, w = cropped.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    return canvas
