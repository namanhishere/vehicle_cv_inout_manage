"""Crop-image path handling and retention pruning.

Paths are built ONLY from generated integers/dates, never from user input.
The ``crop`` column of an event row stores the relative path returned here
(paths under ``images_dir``); viewers join with ``images_dir`` themselves.
"""

from __future__ import annotations

import os
import shutil
from datetime import date, timedelta


def crop_path(images_dir: str, event_id: int) -> str:
    """Relative path ``YYYY/MM/DD/{event_id:08d}_plate.jpg`` (local date)."""
    today = date.today()
    return f"{today:%Y}/{today:%m}/{today:%d}/{int(event_id):08d}_plate.jpg"


def save_crop(images_dir: str, event_id: int, jpeg_bytes: bytes) -> str:
    """Write the crop JPEG under images_dir; returns the relative path."""
    rel = crop_path(images_dir, event_id)
    full = os.path.join(images_dir, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as fh:
        fh.write(jpeg_bytes)
    return rel


def _is_date_dir(path: str) -> date | None:
    """Return the date for a YYYY/MM/DD directory path, or None."""
    parts = path.split(os.sep)
    if len(parts) != 3:
        return None
    try:
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def prune(images_dir: str, retention_days: int) -> None:
    """Delete date directories strictly older than ``retention_days``."""
    if retention_days < 0 or not os.path.isdir(images_dir):
        return
    cutoff = date.today() - timedelta(days=retention_days)
    for year in os.listdir(images_dir):
        ypath = os.path.join(images_dir, year)
        if not os.path.isdir(ypath):
            continue
        for month in os.listdir(ypath):
            mpath = os.path.join(ypath, month)
            if not os.path.isdir(mpath):
                continue
            for day in os.listdir(mpath):
                dpath = os.path.join(mpath, day)
                if not os.path.isdir(dpath):
                    continue
                d = _is_date_dir(os.path.join(year, month, day))
                if d is not None and d < cutoff:
                    shutil.rmtree(dpath, ignore_errors=True)
