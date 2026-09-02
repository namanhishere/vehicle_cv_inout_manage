#!/usr/bin/env python3
"""Generate the committed e2e fixture videos (not binaries in git).

Renders 2-line plates (shared renderer: gate/vision/synth.py) and composes
two videos with plate motion:
  tests/fixtures/e2e_in.mp4    plate 29A1-678.90 (registered in e2e.toml)
  tests/fixtures/e2e_unreg.mp4 plate 29B7-111.11 (unregistered)
Each is 640x360, 10 fps, 10 s: the plate slides across the frame and back.

Usage: .venv/bin/python scripts/make_fixtures.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate.vision.synth import plate_to_array, render_plate  # noqa: E402
from gate.vision.synth import camera_look  # noqa: E402

W, H = 640, 360
FPS = 10
DUR = 10  # seconds
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tests", "fixtures")

# top line, bottom line, output name
SCENES = [
    ("29A1", "678.90", "e2e_in.mp4"),
    ("29B7", "111.11", "e2e_unreg.mp4"),
]


def _scene(plate_img, x0, y0, angle_deg):
    """Plate composited on a dark textured background at (x0, y0).

    Alpha-channel compositing so the rotation preserves dark text pixels
    (a brightness mask would drop them onto the dark background)."""
    rng = np.random.default_rng(42)
    bg = rng.integers(20, 55, size=(H, W, 3), dtype=np.uint8)
    plate = plate_to_array(plate_img)[:, :, ::-1]
    ph, pw = plate.shape[:2]
    scale = 0.6
    pw2, ph2 = int(pw * scale), int(ph * scale)
    plate = cv2.resize(plate, (pw2, ph2), interpolation=cv2.INTER_AREA)
    alpha = np.full((ph2, pw2), 255, dtype=np.uint8)  # plate is opaque
    if angle_deg:
        m = cv2.getRotationMatrix2D((pw2 / 2, ph2 / 2), angle_deg, 1.0)
        plate = cv2.warpAffine(plate, m, (pw2, ph2),
                               borderValue=(40, 40, 40))
        alpha = cv2.warpAffine(alpha, m, (pw2, ph2),
                               borderValue=0, flags=cv2.INTER_LINEAR)
    x0 = max(0, min(int(x0), W - pw2))
    y0 = max(0, min(int(y0), H - ph2))
    roi = bg[y0:y0 + ph2, x0:x0 + pw2]
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    roi[:] = (plate.astype(np.float32) * a
              + roi.astype(np.float32) * (1.0 - a)).astype(np.uint8)
    return camera_look(bg, rng)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    n_frames = DUR * FPS
    for top, bottom, name in SCENES:
        plate_img = render_plate(top, bottom)
        vw = cv2.VideoWriter(
            os.path.join(OUT, name), cv2.VideoWriter_fourcc(*"mp4v"),
            FPS, (W, H),
        )
        # slide in fast (t < 1.2 s: ~25 px/frame -> well above the 2%
        # motion threshold at half resolution), hold (1.2..5 s), slide out
        # (5..6.2 s), gone for the remainder
        for i in range(n_frames):
            t = i / FPS
            if t < 1.2:
                x = int(30 + 300 * t / 1.2)
                angle = 5 * t / 1.2
            elif t < 5:
                x = 330
                angle = 5
            elif t < 6.2:
                x = int(330 + 300 * (t - 5) / 1.2)
                angle = 5 * (6.2 - t) / 1.2
            else:
                x = 700
                angle = 0
            fr = _scene(plate_img, x, 120, angle)
            vw.write(fr)
        vw.release()
        print(f"wrote {OUT}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
