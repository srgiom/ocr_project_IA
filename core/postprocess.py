from __future__ import annotations

from collections import defaultdict
from typing import List

from .segmentation import CharSegment


def rebuild_text_from_segments(segments: List[CharSegment], labels: List[str]) -> str:
    """Reconstruye el texto en líneas/palabras usando las posiciones de los segmentos.

    segments y labels deben ir alineados.
    """
    # Agrupar por línea → palabra
    lines_dict = defaultdict(lambda: defaultdict(list))

    for seg, lab in zip(segments, labels):
        lines_dict[seg.line_idx][seg.word_idx].append((seg.char_idx, lab))

    # Reconstruir en orden
    lines_out = []
    for line_idx in sorted(lines_dict.keys()):
        words_dict = lines_dict[line_idx]
        words_out = []
        for word_idx in sorted(words_dict.keys()):
            chars = sorted(words_dict[word_idx], key=lambda x: x[0])
            word_str = "".join(ch for _, ch in chars)
            words_out.append(word_str)
        lines_out.append(" ".join(words_out))

    return "\n".join(lines_out)
