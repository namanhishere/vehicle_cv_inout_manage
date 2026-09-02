"""Per-side gate controller: motion-gated capture burst -> read -> classify
-> event -> LED.

``SideController.on_frame`` is invoked by the side's ``CameraFeed`` thread
for every frame. Motion is evaluated on a half-resolution gray frame; when
the trigger fires, up to ``burst_frames`` frames are collected, each is run
through the ``PlateReader``, results are grouped by normalized plate, and
the best-supported candidate is classified. Exactly one event row is
written per burst (or none: no detection / cooldown duplicate); ALLOW also
commits the inside/outside transition and saves the event crop; the side's
LEDs signal the outcome.

Invariants: "no detection -> no event" and cooldown suppression live here -
the only place events are written and LEDs fired.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
import time

import cv2
import numpy as np

from gate.plate import normalize

log = logging.getLogger("gate.controller")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SideController:
    """One side (IN or OUT) of the gate."""

    def __init__(
        self,
        side: str,
        camera,
        trigger,
        reader,
        engine,
        db,
        leds,
        storage,
        burst_frames: int = 5,
        min_frames_detected: int = 2,
    ):
        self.side = side
        self.camera = camera
        self.trigger = trigger
        self.reader = reader
        self.engine = engine
        self.db = db
        self.leds = leds
        self.storage = storage
        self.burst_frames = int(burst_frames)
        self.min_frames_detected = int(min_frames_detected)
        self._burst: list[np.ndarray] | None = None
        self._lock = threading.Lock()
        # stats for the dashboard/watchdog (thread-safe counters)
        self.frames_seen = 0
        self.last_burst_at = 0.0

    # -- per-frame entry ---------------------------------------------------

    def on_frame(self, frame_bgr: np.ndarray) -> None:
        """Camera-thread callback: motion gate + burst collection."""
        self.frames_seen += 1
        small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        with self._lock:
            if self._burst is not None:
                self._burst.append(frame_bgr)
                if len(self._burst) >= self.burst_frames:
                    burst = self._burst
                    self._burst = None
                    self.last_burst_at = time.time()
                else:
                    return
            elif self.trigger.update(gray):
                self._burst = [frame_bgr]
                return
            else:
                return
        # full pipeline outside the lock
        try:
            self._process_burst(burst)
        except Exception:
            log.exception("side %s burst processing failed", self.side)

    # -- burst processing --------------------------------------------------

    def _process_burst(self, burst: list[np.ndarray]) -> None:
        if len(burst) < 2:
            return  # a valid vote needs >= 2 frames
        groups: dict[str, dict] = {}
        for frame in burst:
            result = self.reader.read(frame)
            if result is None:
                continue
            raw, conf = result
            parsed = normalize(raw)
            key = parsed.canonical if parsed is not None else raw
            group = groups.setdefault(
                key,
                {"count": 0, "conf_sum": 0.0, "best_conf": -1.0,
                 "best_raw": "", "best_frame": None, "valid": parsed},
            )
            group["count"] += 1
            group["conf_sum"] += conf
            if conf > group["best_conf"]:
                group["best_conf"] = conf
                group["best_raw"] = raw
                group["best_frame"] = frame
        accepted = [
            g for g in groups.values()
            if g["count"] >= self.min_frames_detected
        ]
        if not accepted:
            return  # no stable reading: no event, no LED
        # highest support; ties broken by mean confidence
        chosen = max(
            accepted,
            key=lambda g: (g["count"], g["conf_sum"] / g["count"]),
        )
        confidence = chosen["conf_sum"] / chosen["count"]
        decision = self.engine.classify(
            raw=chosen["best_raw"],
            direction=self.side,
            confidence=confidence,
            camera=self.side,
        )
        if decision.duplicate:
            return  # cooldown dedupe: no event, no LED, no state change
        event_id = self.db.record_event(
            ts=_now_iso(),
            plate=decision.plate,
            raw=chosen["best_raw"],
            direction=self.side,
            result=decision.result.value,
            reason=decision.reason.value,
            confidence=confidence,
            camera=self.side,
            crop=None,
        )
        if decision.result.value == "ALLOW":
            self.engine.apply(decision)
            self._save_crop(event_id, chosen["best_frame"])
            outcome = "ALLOW"
        elif decision.reason.value == "LOW_CONF":
            outcome = "LOW_CONF"
        else:
            outcome = "REJECT"
        self.leds.signal(self.side, outcome, confidence)

    def _save_crop(self, event_id: int, frame_bgr: np.ndarray | None) -> None:
        """Persist the best plate crop for an ALLOW event (best effort)."""
        if frame_bgr is None:
            return
        try:
            crops = self.reader.detector.detect(frame_bgr)
            if not crops:
                return
            ok, jpeg = cv2.imencode(
                ".jpg", crops[0], [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not ok:
                return
            rel = self.storage.save_crop(event_id, jpeg.tobytes())
            self.db.update_crop(event_id, rel)
        except Exception:
            log.exception("side %s crop save failed", self.side)

    # -- lifecycle ---------------------------------------------------------

    def stop(self) -> None:
        """Called on shutdown; the camera feed owns its own thread."""
        with self._lock:
            self._burst = None
        self.leds.all_off()
