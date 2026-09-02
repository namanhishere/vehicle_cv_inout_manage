"""Plate reading pipeline: detect -> split lines -> OCR both lines.

Assembles the two line readings into one raw string (separators are handled
by ``plate.normalize`` downstream) and returns the highest mean-confidence
candidate, or None when nothing could be read.
"""

from __future__ import annotations

import numpy as np

from gate.vision.lines import split_lines


class PlateReader:
    def __init__(self, detector, ocr):
        self.detector = detector
        self.ocr = ocr

    def read(self, frame_bgr: np.ndarray) -> tuple[str, float] | None:
        """Return (assembled_raw, mean_conf) for the best candidate or None."""
        best: tuple[str, float] | None = None
        for crop in self.detector.detect(frame_bgr):
            lines = split_lines(crop)
            if lines is None:
                continue
            top, bottom = lines
            t_text, t_conf = self.ocr.recognize(top)
            b_text, b_conf = self.ocr.recognize(bottom)
            if not t_text or not b_text:
                continue
            conf = (t_conf + b_conf) / 2.0
            if best is None or conf > best[1]:
                best = (t_text + b_text, conf)
        return best
