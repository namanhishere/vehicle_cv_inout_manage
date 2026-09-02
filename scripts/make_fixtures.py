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
    """Plate composited on a dark textured background at (x0, y0)."""
    rng = np.random.default_rng(42)
    bg = rng.integers(20, 55, size=(H, W, 3), dtype=np.uint8)
    plate = plate_to_array(plate_img)[:, :, ::-1]
    ph, pw = plate.shape[:2]
    scale = 0.5
    pw2, ph2 = int(pw * scale), int(ph * scale)
    plate = cv2.resize(plate, (pw2, ph2), interpolation=cv2.INTER_AREA)
    if angle_deg:
        m = cv2.getRotationMatrix2D((pw2 / 2, ph2 / 2), angle_deg, 1.0)
        plate = cv2.warpAffine(plate, m, (pw2, ph2),
                               borderValue=(30, 30, 30))
    mask = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) > 40
    x0, y0 = int(x0), int(y0)
    roi = bg[y0:y0 + ph2, x0:x0 + pw2]
    roi[mask] = plate[mask]
    return bg


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    n_frames = DUR * FPS
    for top, bottom, name in SCENES:
        plate_img = render_plate(top, bottom)
        vw = cv2.VideoWriter(
            os.path.join(OUT, name), cv2.VideoWriter_fourcc(*"mp4v"),
            FPS, (W, H),
        )
        # slide in (0..2 s), hold (2..6 s), slide out (6..9 s)
        for i in range(n_frames):
            t = i / FPS
            if t < 2:
                x = int(40 + 300 * t / 2)
                angle = 6 * t
            elif t < 6:
                x = 340
                angle = 6
            elif t < 9:
                x = int(340 + 300 * (t - 6) / 3)
                angle = 6 * (9 - t) / 3
            else:
                x = 640
                angle = 0
            fr = _scene(plate_img, x, 130, angle)
            vw.write(fr)
        vw.release()
        print(f"wrote {OUT}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
