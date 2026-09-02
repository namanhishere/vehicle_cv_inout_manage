"""Tests for gate/display.py - runs under SDL_VIDEODRIVER=dummy."""

import os
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"  # before pygame import

import pytest

from gate.db import GateDB
from gate.display import StatusDisplay
from gate.storage import Storage

PLATE = "29A1-678.90"


class FakeState:
    def __init__(self, ok=True):
        self.ok = ok

    def camera_ok(self, side):
        return self.ok


@pytest.fixture
def db(tmp_path):
    d = GateDB(str(tmp_path / "gate.db"))
    yield d
    d.close()


def test_renders_frames_without_exception(db, tmp_path):
    disp = StatusDisplay(db, FakeState(), width=320, height=240)
    disp.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and disp.screen is None:
        time.sleep(0.05)
    assert disp.screen is not None
    time.sleep(0.45)  # >= 3 frames at 10 fps
    disp.stop()
    assert not disp.is_alive()


def test_screen_changes_after_event_insert(db, tmp_path):
    disp = StatusDisplay(db, FakeState(), width=320, height=240)
    disp.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and disp.screen is None:
        time.sleep(0.05)
    assert disp.screen is not None
    time.sleep(0.3)
    shot_a = tmp_path / "a.png"
    pygame_image_save(disp.screen, str(shot_a))
    db.record_event(
        ts="2026-01-01T12:00:00", plate=PLATE, raw=PLATE, direction="IN",
        result="ALLOW", reason="ALLOW", confidence=0.9, camera="IN",
        crop=None,
    )
    time.sleep(0.3)
    shot_b = tmp_path / "b.png"
    pygame_image_save(disp.screen, str(shot_b))
    disp.stop()

    import pygame

    a = pygame.image.load(str(shot_a))
    b = pygame.image.load(str(shot_b))
    wa, ha = a.get_size()
    diff = sum(
        1
        for x in range(0, wa, 7)
        for y in range(0, ha, 7)
        if a.get_at((x, y)) != b.get_at((x, y))
    )
    assert diff > 0, "screen should change after an event insert"
    # and the later frame is not all black (text/boxes drawn)
    colors = {
        tuple(b.get_at((x, y)))
        for x in range(0, wa, 5)
        for y in range(0, ha, 5)
    }
    assert colors != {(0, 0, 0, 255)}


def pygame_image_save(surface, path):
    import pygame

    pygame.image.save(surface, path)


def test_camera_fail_status_shown(db, tmp_path):
    disp = StatusDisplay(db, FakeState(ok=False), width=320, height=240)
    disp.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and disp.screen is None:
        time.sleep(0.05)
    assert disp.screen is not None
    time.sleep(0.3)
    disp.stop()
    # exercised the CAMERA FAIL path without exception
