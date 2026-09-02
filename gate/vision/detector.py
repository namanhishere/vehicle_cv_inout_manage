"""YOLOv8n plate detector decode on plain onnxruntime (no ultralytics
runtime dependency on the Pi).

Preprocess: letterbox frame to 640x640 (pad gray 114), BGR->RGB, /255,
HWC->CHW, NCHW float32. Output tensor [1, 4+nc, 8400]: (cx, cy, w, h) +
class scores. The exporter may emit [1, 8400, 4+nc]; the layout is read at
runtime and normalized.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger("gate.vision.detector")

_SESSIONS: dict[str, ort.InferenceSession] = {}  # lazy per-process singleton


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


class YoloPlateDetector:
    """Detects motorcycle plates; returns crops (BGR uint8), best first."""

    def __init__(self, onnx_path: str, conf: float = 0.25, iou: float = 0.45):
        self.onnx_path = onnx_path
        self.conf = conf
        self.iou = iou
        self._sess = None  # lazy
        self._input_name = None
        self._input_h = self._input_w = 640

    # -- inference ---------------------------------------------------------

    def _ensure(self) -> ort.InferenceSession:
        if self._sess is None:
            self._sess = _session(self.onnx_path)
            inp = self._sess.get_inputs()[0]
            self._input_name = inp.name
            shape = inp.shape
            if len(shape) == 4 and isinstance(shape[2], int):
                self._input_h, self._input_w = int(shape[2]), int(shape[3])
        return self._sess

    def detect(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        """Return up to 3 plate crops (BGR uint8), best confidence first;
        [] when nothing passes the confidence threshold."""
        sess = self._ensure()
        ih, iw = self._input_h, self._input_w
        h, w = frame_bgr.shape[:2]
        scale = min(iw / w, ih / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        pad_x = (iw - nw) // 2
        pad_y = (ih - nh) // 2

        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        out = sess.run(None, {self._input_name: blob})[0]
        arr = np.squeeze(out, axis=0)
        if arr.shape[0] != 5:
            arr = arr.T  # exporter put channels last

        boxes_xywh = arr[:4].T  # (N, 4) in letterboxed pixels
        scores = arr[4:].max(axis=0)
        keep = scores >= self.conf
        if not keep.any():
            return []
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]

        order = np.argsort(-scores)
        boxes_xywh = boxes_xywh[order]
        scores = scores[order]

        # map back to original frame coordinates
        orig = []
        for (cx, cy, bw, bh) in boxes_xywh:
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale
            orig.append((x1, y1, x2, y2))
        kept = cv2.dnn.NMSBoxes(
            [(float(x1), float(y1), float(x2 - x1), float(y2 - y1))
             for (x1, y1, x2, y2) in orig],
            scores.tolist(),
            self.conf,
            self.iou,
        )
        if kept is None or len(kept) == 0:
            return []
        kept = np.asarray(kept).ravel()  # cv2 may return (N,1) or (N,)
        crops = []
        for idx in kept[:3]:
            x1, y1, x2, y2 = orig[int(idx)]
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(w - 1, int(round(x2)))
            y2 = min(h - 1, int(round(y2)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crops.append(frame_bgr[y1:y2 + 1, x1:x2 + 1])
        return crops
