from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from config import MIN_COMPONENT_AREA, LINE_MERGE_THRESHOLD, WORD_GAP_FACTOR
from .preprocessing import normalize_char_image


@dataclass
class CharSegment:
    char_image: np.ndarray
    x: int
    y: int
    w: int
    h: int
    line_idx: int
    word_idx: int
    char_idx: int


def _find_connected_components(binary: np.ndarray):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # stats: [label, x, y, w, h, area]
    components = []
    for label in range(1, num_labels):  # saltar fondo
        x, y, w, h, area = stats[label]
        if area < MIN_COMPONENT_AREA:
            continue
        cx, cy = centroids[label]
        components.append(
            {
                "label": label,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "cx": cx,
                "cy": cy,
            }
        )
    return components


def _group_components_into_lines(components):
    """Agrupa componentes en líneas basadas en la coordenada y del centro y la altura media."""
    if not components:
        return []

    comps_sorted = sorted(components, key=lambda c: c["cy"])
    lines: list[list[dict]] = []

    for comp in comps_sorted:
        placed = False
        for line in lines:
            # Comparar con la línea usando el promedio de alturas
            line_ys = [c["cy"] for c in line]
            line_heights = [c["h"] for c in line]
            mean_y = np.mean(line_ys)
            mean_h = np.mean(line_heights)

            if abs(comp["cy"] - mean_y) <= LINE_MERGE_THRESHOLD * mean_h:
                line.append(comp)
                placed = True
                break

        if not placed:
            lines.append([comp])

    # Ordenar cada línea por x
    for line in lines:
        line.sort(key=lambda c: c["x"])
    return lines


def _split_line_into_words(line_components):
    """Divide componentes de una línea en palabras según gaps horizontales."""
    if len(line_components) == 0:
        return []

    # Calcular gaps entre cajas consecutivas
    gaps = []
    for i in range(len(line_components) - 1):
        c1 = line_components[i]
        c2 = line_components[i + 1]
        gap = c2["x"] - (c1["x"] + c1["w"])
        gaps.append(max(0, gap))

    if gaps:
        mean_gap = np.mean(gaps)
    else:
        mean_gap = 0

    words = []
    current_word = [line_components[0]]

    for i in range(len(line_components) - 1):
        c1 = line_components[i]
        c2 = line_components[i + 1]
        gap = c2["x"] - (c1["x"] + c1["w"])
        if mean_gap > 0 and gap > WORD_GAP_FACTOR * mean_gap:
            # Nuevo espacio
            words.append(current_word)
            current_word = [c2]
        else:
            current_word.append(c2)

    words.append(current_word)
    return words


def segment_characters_from_binary(binary: np.ndarray) -> List[CharSegment]:
    """Segmenta la imagen binaria en caracteres individuales.

    Devuelve una lista de CharSegment con la imagen normalizada del carácter
    y su posición (para poder reconstruir el texto por líneas/palabras).
    """
    components = _find_connected_components(binary)
    lines = _group_components_into_lines(components)

    segments: List[CharSegment] = []
    for line_idx, line in enumerate(lines):
        words = _split_line_into_words(line)
        for word_idx, word in enumerate(words):
            for char_idx, comp in enumerate(word):
                x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
                char_crop = binary[y : y + h, x : x + w]
                norm_char = normalize_char_image(char_crop)
                segments.append(
                    CharSegment(
                        char_image=norm_char,
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        line_idx=line_idx,
                        word_idx=word_idx,
                        char_idx=char_idx,
                    )
                )
    return segments
