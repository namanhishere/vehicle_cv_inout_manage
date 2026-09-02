"""Camera capture thread with auto-recovery, plus motion trigger.

``CameraFeed`` reads a V4L2 index or RTSP/file source in a daemon thread and
invokes ``on_frame(copy)`` for every frame. Any read failure releases and
reopens the source with a backoff, retrying forever; nothing escapes the
loop. ``ok`` reflects whether the last read succeeded, ``last_frame_at`` the
monotonic time of the last successful read (watchdog consumes both).

``MotionTrigger`` fires when the changed-pixel fraction between consecutive
gray frames exceeds a threshold, debounced (after a trigger, further
triggers are suppressed for ``debounce_s``).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger("gate.capture")


class CameraFeed(threading.Thread):
    """Reads frames from a camera source and pushes copies to ``on_frame``."""

    def __init__(
        self,
        name: str,
        source: str,
        width: int,
        height: int,
        fps: int,
        on_frame: Callable[[np.ndarray], None],
        retry_delay_s: float = 1.0,
    ):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.name_tag = name
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.on_frame = on_frame
        self.retry_delay_s = retry_delay_s
        self._stop_event = threading.Event()
        self._ok = False
        self._last_frame_at = 0.0
        self._lock = threading.Lock()

    # -- properties --------------------------------------------------------

    @property
    def ok(self) -> bool:
        with self._lock:
            return self._ok

    @property
    def last_frame_at(self) -> float:
        with self._lock:
            return self._last_frame_at

    # -- lifecycle ---------------------------------------------------------

    def _open(self):
        """Open the source or return None (never raises)."""
        try:
            if str(self.source).isdigit():
                cap = cv2.VideoCapture(int(self.source))
            else:
                cap = cv2.VideoCapture(str(self.source))
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            return cap
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("cam %s open error: %s", self.name_tag, exc)
            return None

    def run(self) -> None:
        while not self._stop_event.is_set():
            cap = self._open()
            if cap is None:
                log.warning(
                    "cam %s: source %r unavailable; retrying in %ss",
                    self.name_tag, self.source, self.retry_delay_s,
                )
                self._set_ok(False)
                self._stop_event.wait(self.retry_delay_s)
                continue
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                self._set_ok(True)
                with self._lock:
                    self._last_frame_at = time.monotonic()
                try:
                    self.on_frame(frame.copy())
                except Exception:
                    log.exception("cam %s on_frame handler failed", self.name_tag)
            cap.release()
            self._set_ok(False)
            if not self._stop_event.is_set():
                self._stop_event.wait(self.retry_delay_s)

    def _set_ok(self, ok: bool) -> None:
        with self._lock:
            self._ok = ok

    def stop(self) -> None:
        """Stop the loop and join the thread."""
        self._stop_event.set()
        self.join(timeout=5.0)


class MotionTrigger:
    """Fires once per motion event, debounced; cheap (gray frames only)."""

    def __init__(
        self,
        threshold: float = 0.02,
        debounce_s: float = 3.0,
        now: Callable[[], float] = time.monotonic,
    ):
        self.threshold = float(threshold)
        self.debounce_s = float(debounce_s)
        self._prev: np.ndarray | None = None
        self._last_trigger = float("-inf")
        self._now = now

    def update(self, frame_gray: np.ndarray) -> bool:
        """Feed one grayscale frame; True when a motion event starts."""
        now = self._now()
        prev = self._prev
        self._prev = frame_gray
        if prev is None or prev.shape != frame_gray.shape:
            return False
        diff = cv2.absdiff(prev, frame_gray)
        _t, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        fraction = cv2.countNonZero(binary) / float(frame_gray.size)
        if fraction > self.threshold and now - self._last_trigger >= self.debounce_s:
            self._last_trigger = now
            return True
        return False
