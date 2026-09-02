"""Split a motorcycle plate crop into its two text lines.

Motorcycle plates are 2-line, 190x140 mm. Grayscale -> Otsu binary ->
horizontal projection; the line separator is the widest text-free valley
inside the middle zone of the crop (text lines never span the plate's
middle). A single text band (1-line plate, e.g. a car plate) leaves one
side of any middle split empty -> None: the motorcycle gate treats it as a
no-read and the policy rejects car plates.

Each returned band is padded 4 px. Frame lines and borders are irrelevant
to this scheme (they live outside the middle zone or attach to a text
band's own line region).
"""

from __future__ import annotations

import cv2
import numpy as np


def _row_activity(binary: np.ndarray) -> np.ndarray:
    """Active (text-bearing) rows; side margins excluded so a plate's thin
    vertical frame lines (present in every row) never glue bands together."""
    h, w = binary.shape
    margin = max(2, int(0.04 * w))
    inner = binary[:, margin:w - margin] if w > 2 * margin else binary
    return (inner.sum(axis=1) / 255.0) > max(3.0, 0.02 * inner.shape[1])


def split_lines(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (top_line, bottom_line) BGR crops, or None for 1-line crops."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    active = _row_activity(binary)
    h = crop_bgr.shape[0]

    # widest inactive run strictly inside the middle zone. Each text line
    # occupies roughly a third of the plate, so the inter-line gap always
    # falls inside [0.3h, 0.7h]; top margins/border smear live above it.
    lo, hi = int(0.3 * h), int(0.7 * h)
    best_start, best_len = -1, 0
    run = 0
    for y in range(lo, hi):
        if not active[y]:
            run += 1
            if run > best_len:
                best_len, best_start = run, y - run + 1
        else:
            run = 0
    if best_len < 2:
        return None
    split = best_start + best_len // 2

    top_idx = np.where(active[:split])[0]
    bot_idx = np.where(active[split:])[0] + split
    if len(top_idx) == 0 or len(bot_idx) == 0:
        return None  # one-line plate (car) or margin-only valley
    # a real text line occupies a big share of the plate; reject slivers
    # (frame lines etc.) masquerading as a line
    min_span = max(6, int(0.08 * h))
    if (int(top_idx.max()) - int(top_idx.min()) < min_span
            or int(bot_idx.max()) - int(bot_idx.min()) < min_span):
        return None
    pad = 4
    top = crop_bgr[max(0, int(top_idx.min()) - pad):int(top_idx.max()) + 1 + pad, :]
    bot = crop_bgr[int(bot_idx.min()) - pad:min(h, int(bot_idx.max()) + 1 + pad), :]
    return top, bot
