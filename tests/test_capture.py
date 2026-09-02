"""Tests for gate/capture.py (CameraFeed) and gate/motion.py.

Synthetic videos are composed at test time with numpy + cv2.VideoWriter
(ffmpeg drawbox with t-expressions misbehaves on lavfi color input).
"""

import time

import cv2
import numpy as np
import pytest

from gate.capture import CameraFeed, MotionTrigger

W, H = 640, 360
FPS = 10
DUR_FRAMES = 50  # 5 s


def _compose(path, motion):
    """Write gray frames; with motion, a 140x120 white box appears at frame
    10 and sweeps right at 30 px/frame (per-frame change ~3.1%, above the
    2% threshold) until it exits around frame 27."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                         (W, H))
    for i in range(DUR_FRAMES):
        fr = np.full((H, W, 3), 128, dtype=np.uint8)
        if motion and 10 <= i < 27:
            x = int(30 * (i - 10))
            fr[120:240, x:x + 140] = 255
        vw.write(fr)
    vw.release()


@pytest.fixture
def videos(tmp_path):
    """Build a static and a motion video (640x360, 10 fps, 5 s)."""
    static = tmp_path / "static.mp4"
    motion = tmp_path / "motion.mp4"
    _compose(static, motion=False)
    _compose(motion, motion=True)
    return static, motion


def frames_of(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames


class FeedClock:
    """Drives MotionTrigger at a simulated 10 fps cadence."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def tick(self):
        self.t += 0.1


def run_trigger(trig, clock, frames):
    hits = 0
    for fr in frames:
        clock.tick()
        if trig.update(fr):
            hits += 1
    return hits


def test_camera_feed_reads_file(videos):
    static, _ = videos
    counts = {"n": 0, "ok_seen": False}

    def on_frame(frame):
        counts["n"] += 1
        counts["ok_seen"] = counts["ok_seen"] or feed.ok

    feed = CameraFeed("in", str(static), W, H, 10, on_frame,
                      retry_delay_s=0.05)
    feed.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and counts["n"] < 20:
        time.sleep(0.05)
    assert counts["n"] >= 20
    # ok was True during delivery (checked inside on_frame, avoiding the
    # EOF-reopen race on the short test file)
    assert counts["ok_seen"]
    feed.stop()
    assert not feed.is_alive()


def test_camera_feed_corrupt_file_never_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video file at all")

    def on_frame(frame):  # pragma: no cover - must never be called
        raise AssertionError("no frames from a corrupt source")

    feed = CameraFeed("in", str(bad), W, H, 10, on_frame,
                      retry_delay_s=0.05)
    feed.start()
    time.sleep(0.5)  # let it attempt several open/retry cycles
    assert not feed.ok
    assert feed.is_alive()  # keeps looping, does not raise
    feed.stop()
    assert not feed.is_alive()


def test_motion_trigger_fires_on_motion_only(videos):
    static, motion = videos
    clock = FeedClock()
    hits = run_trigger(MotionTrigger(threshold=0.02, debounce_s=3.0,
                                     now=clock), clock, frames_of(motion))
    assert hits >= 1  # box sweep window triggers

    clock2 = FeedClock()
    hits_static = run_trigger(MotionTrigger(threshold=0.02, debounce_s=0.0,
                                            now=clock2), clock2,
                              frames_of(static))
    assert hits_static == 0


def test_motion_trigger_debounce_holds(videos):
    _, motion = videos
    # debounce longer than the motion window: exactly one trigger
    clock = FeedClock()
    hits = run_trigger(MotionTrigger(threshold=0.02, debounce_s=5.0,
                                     now=clock), clock, frames_of(motion))
    assert hits == 1
    # short debounce: re-triggers while the box is still sweeping
    clock2 = FeedClock()
    hits2 = run_trigger(MotionTrigger(threshold=0.02, debounce_s=0.3,
                                      now=clock2), clock2, frames_of(motion))
    assert hits2 >= 2


def test_motion_trigger_first_frame_no_fire():
    trig = MotionTrigger(debounce_s=0.0)
    assert not trig.update(np.zeros((180, 320), dtype=np.uint8))
    assert not trig.update(np.zeros((180, 320), dtype=np.uint8))
    noisy = np.zeros((180, 320), dtype=np.uint8)
    noisy[50:80, 50:200] = 255
    assert trig.update(noisy)
