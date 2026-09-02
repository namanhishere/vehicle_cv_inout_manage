"""Synthetic Vietnamese plate rendering for tests and fixture generation.

Realistic 2-line plates (white plate, black text, 190x140 mm aspect) are
rendered with Pillow + DejaVu fonts and pasted onto noise/gradient
backgrounds with perspective warp and JPEG compression. Shared by the
pytest conftest and scripts/make_fixtures.py.

No dataset images are ever redistributed; everything here is generated.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Plate geometry: real plates are 190x140 mm -> 1.357:1; two text lines.
PLATE_W, PLATE_H = 380, 280
LINE1_Y, LINE2_Y = 62, 168  # top text line / bottom text line baseline area
FONT_SZ = 92


def _load_font(size: int):
    return ImageFont.truetype(FONT_PATH, size)


def render_plate(top_text: str, bottom_text: str, scale: float = 1.0) -> Image.Image:
    """White 2-line plate image; returns a PIL RGB image."""
    w = int(PLATE_W * scale)
    h = int(PLATE_H * scale)
    img = Image.new("RGB", (w, h), (246, 246, 244))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, w - 3, h - 3], outline=(40, 80, 160), width=max(2, int(4 * scale)))
    f = _load_font(int(FONT_SZ * scale))
    for text, y in ((top_text, LINE1_Y), (bottom_text, LINE2_Y)):
        draw.text((24, int(y * scale)), text, fill=(18, 18, 20), font=f)
    return img


def plate_to_array(plate_img: Image.Image) -> np.ndarray:
    return np.asarray(plate_img.convert("RGB"))


def camera_look(bgr: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Make a composited frame look like camera footage.

    The plate detector was trained on real photographs; pristine flat
    renders are out-of-distribution for it. Mild blur + sensor noise +
    brightness/gain compression bridge the gap (kept light enough that
    OCR still reads the plate)."""
    if rng is None:
        rng = np.random.default_rng(7)
    # (blur 1.0, gain 0.8, offset 15, noise 4) chosen so the plate detector
    # scores composite renders well above its threshold while OCR keeps
    # enough glyph sharpness for high-confidence reads
    img = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    noise = rng.normal(0, 4.0, size=img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255)
    img = np.clip(img * 0.8 + 15.0, 0, 255)
    return img.astype(np.uint8)


def composite_scene(
    plate_img: Image.Image,
    canvas: tuple[int, int] = (640, 360),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Paste a perspective-warped plate onto a noisy background (BGR ndarray)."""
    if rng is None:
        rng = np.random.default_rng(7)
    cw, ch = canvas
    bg = rng.integers(0, 60, size=(ch, cw, 3), dtype=np.uint8)
    grad = np.linspace(0, 90, ch, dtype=np.uint8)[:, None, None]
    bg = np.clip(bg.astype(np.int16) + grad, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 14, size=(ch, cw, 3))
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    pw, ph = plate_img.size
    # ~0.38 of the frame width: matches the detector's training scale,
    # where boxes come out tight (larger synthetic plates yield loose,
    # background-including boxes)
    max_w = int(cw * 0.45)
    scale = max_w / pw
    sw, sh = int(pw * scale), int(ph * scale)
    plate_img = plate_img.resize((sw, sh), Image.LANCZOS)

    # perspective warp: perturb the four corners a little
    k = 0.05
    dx = lambda: int(rng.uniform(-k * sw, k * sw))  # noqa: E731
    dy = lambda: int(rng.uniform(-k * sh, k * sh))  # noqa: E731
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    dst = np.float32([[dx(), dy()], [sw + dx(), dy()],
                      [sw + dx(), sh + dy()], [dx(), sh + dy()]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate_to_array(plate_img), m, (sw, sh))
    warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)

    x0 = rng.integers(30, max(31, cw - sw - 30))
    y0 = rng.integers(30, max(31, ch - sh - 30))
    scene = bg.copy()
    roi = scene[y0:y0 + sh, x0:x0 + sw]
    mask = np.all(warped > 20, axis=2)
    roi[mask] = warped[mask]
    scene = camera_look(scene, rng)
    # JPEG-style compression artifacts
    ok, enc = cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, 82])
    assert ok
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)
