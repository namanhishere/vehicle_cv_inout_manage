"""Shared synthetic-plate fixtures for vision/controller/e2e tests.

Plates are rendered at test time (Pillow + DejaVu) - no dataset images are
redistributed. Deterministic per (seed, text) via the generator parameter.
"""

from __future__ import annotations

import numpy as np
import pytest

from gate.vision.synth import composite_scene, render_plate


def scene_for(top_text: str, bottom_text: str, seed: int = 7) -> np.ndarray:
    """BGR composite scene (640x360) with the plate somewhere in it."""
    rng = np.random.default_rng(seed)
    plate = render_plate(top_text, bottom_text)
    return composite_scene(plate, rng=rng)


def plate_crop_bgr(top_text: str, bottom_text: str) -> np.ndarray:
    """Clean plate crop (BGR ndarray) without background compositing."""
    return np.asarray(render_plate(top_text, bottom_text).convert("RGB"))[
        :, :, ::-1
    ]


@pytest.fixture(scope="session")
def make_scene():
    """Factory: make_scene(top, bottom, seed=7) -> BGR composite scene."""
    return scene_for
