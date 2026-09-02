"""PP-OCRv3 English recognition (rec) inference via ONNX.

Preprocess per PaddleOCR rec spec: grayscale, resize to height 48 keeping
aspect ratio (width = floor8 multiples, capped at 320), normalize with
mean/std 0.5, transpose to CHW, NCHW float32. Decode: argmax per timestep,
CTC-style collapse (drop repeats and the blank class), map through the
model dictionary; confidence = mean of the decoded characters' max probs.

Dictionary: the model outputs 97 classes. ``models/vendor`` ships no dict
file and PaddleOCR's 37-entry ``en_dict.txt`` does not match this model, so
the layout below was recovered empirically by glyph probing:

- class 0 = CTC blank (a blank image yields all-0 timesteps)
- classes 1..10  = '0'..'9'
- classes 18..43 = 'A'..'Z'  (A->18, B->19, ..., verified runs W/X/Y -> 40/41/42)
- classes 50..75 = 'a'..'z'  (a->50 ... z->75; uppercase glyphs frequently
  decode to their lowercase class - plate.normalize() uppercases anyway)
- classes 11..17, 44..49, 76..96 = punctuation/accents; plates never
  contain them, so they decode to '' (dropped from the raw text).

Note: 'D' with the Vietnamese horn (D) probes as plain 'D'/'d' (class 21/53).
The MD/MD disambiguation in the decision engine resolves that ambiguity
against the registry, so this is expected behavior, not a bug.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

_BLANK = "<blank>"
_UNKNOWN = ""  # classes that never occur on plates

# 1 blank + 10 digits + 7 punct + 26 upper + 6 punct + 26 lower + 21 extra
_CHARS: list[str] = (
    [_BLANK]
    + list("0123456789")
    + [_UNKNOWN] * 7
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + [_UNKNOWN] * 6
    + list("abcdefghijklmnopqrstuvwxyz")
    + [_UNKNOWN] * 21
)
assert len(_CHARS) == 97

_SESSIONS: dict[str, ort.InferenceSession] = {}


def _session(path: str) -> ort.InferenceSession:
    sess = _SESSIONS.get(path)
    if sess is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        _SESSIONS[path] = sess
    return sess


class OcrEngine:
    """PP-OCRv3 en rec on one line crop."""

    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self._sess = None
        self._input_name = None
        self._height = 48

    def _ensure(self):
        if self._sess is None:
            sess = _session(self.onnx_path)
            self._sess = sess
            self._input_name = sess.get_inputs()[0].name
        return self._sess

    def recognize(self, line_bgr: np.ndarray) -> tuple[str, float]:
        """(decoded text, mean char probability)."""
        sess = self._ensure()
        gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = self._height / h
        rw = int(round(w * scale))
        rw = max(8, (rw // 8) * 8)
        rw = min(320, rw)
        if rw < 8:
            return "", 0.0
        resized = cv2.resize(gray, (rw, self._height), interpolation=cv2.INTER_LINEAR)
        img = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        blob = np.repeat(img[np.newaxis, np.newaxis, :, :], 3, axis=1).astype(
            np.float32
        )
        logits = sess.run(None, {self._input_name: blob})[0][0]  # [T, 97]
        ids = logits.argmax(axis=1)
        probs = logits.max(axis=1)

        chars: list[str] = []
        probs_of_chars: list[float] = []
        last = -1  # last non-blank class id
        for t in range(ids.shape[0]):
            c = int(ids[t])
            if c == 0:  # blank: resets repeat-collapse context
                last = -1
                continue
            if c == last:  # collapse only *adjacent* repeats
                continue
            last = c
            ch = _CHARS[c]
            if ch == _UNKNOWN:
                continue
            chars.append(ch)
            probs_of_chars.append(float(probs[t]))
        text = "".join(chars)
        confidence = sum(probs_of_chars) / len(probs_of_chars) if probs_of_chars else 0.0
        return text, confidence
