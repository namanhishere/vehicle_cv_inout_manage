#!/usr/bin/env python3
"""VN Plate Gate entrypoint.

Sequence: load config -> GateDB -> shared detector/OCR singletons (both
sides share one reader - memory bound) -> IN/OUT SideControllers with their
own CameraFeed threads -> waitress web thread -> optional framebuffer
display (--display) -> watchdog thread (camera liveness, DB ping; it NEVER
kills the process - systemd owns restart). SIGINT/SIGTERM stop gracefully.

Degraded start: with no cameras the feeds log and retry forever; the web
UI still serves and the dashboard shows camera FAIL.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time

from gate.capture import CameraFeed, MotionTrigger
from gate.config import Config, load_config
from gate.controller import SideController
from gate.db import GateDB
from gate.decision import DecisionEngine
from gate.display import StatusDisplay
from gate.leds import GpioLedController
from gate.storage import Storage
from gate.vision.detector import YoloPlateDetector
from gate.vision.ocr import OcrEngine
from gate.vision.pipeline import PlateReader
from gate.web.app import create_app

log = logging.getLogger("gate.app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

WATCHDOG_S = 5.0


class RuntimeState:
    """Live status surface consumed by the web UI and the display."""

    def __init__(self):
        self._feeds: dict[str, CameraFeed] = {}

    def attach(self, side: str, feed: CameraFeed) -> None:
        self._feeds[side] = feed

    def camera_ok(self, side: str) -> bool:
        feed = self._feeds.get(side)
        return feed is not None and feed.ok


class GateApp:
    def __init__(self, config: Config, with_display: bool = False):
        self.config = config
        self.with_display = with_display
        self.db = GateDB(config.storage.db_path)
        self.storage = Storage(config.storage.images_dir)
        model_dir = config.vision.model_dir
        self.reader = PlateReader(
            YoloPlateDetector(os.path.join(model_dir, "plate_det.onnx")),
            OcrEngine(os.path.join(model_dir, "ocr_rec.onnx")),
        )
        self.engine = DecisionEngine(
            self.db, config.vision.min_confidence,
            config.decision.cooldown_s,
        )
        pins = {
            "in_green": config.leds.in_green,
            "in_red": config.leds.in_red,
            "out_green": config.leds.out_green,
            "out_red": config.leds.out_red,
        }
        self.leds = GpioLedController(
            pins, config.leds.allow_s, config.leds.reject_s,
            config.leds.blink_s,
        )
        self.state = RuntimeState()
        self.feeds: dict[str, CameraFeed] = {}
        self.controllers: dict[str, SideController] = {}
        for side in ("IN", "OUT"):
            cam = config.cameras[side]
            trigger = MotionTrigger()
            ctrl = SideController(
                side, camera=None, trigger=trigger, reader=self.reader,
                engine=self.engine, db=self.db, leds=self.leds,
                storage=self.storage,
                burst_frames=config.vision.burst_frames,
                min_frames_detected=config.vision.min_frames_detected,
            )
            feed = CameraFeed(side, cam.source, cam.width, cam.height,
                              cam.fps, ctrl.on_frame)
            ctrl.camera = feed
            self.feeds[side] = feed
            self.controllers[side] = ctrl
            self.state.attach(side, feed)
        self._stop = threading.Event()
        self.web_thread: threading.Thread | None = None
        self.display: StatusDisplay | None = None
        self.watchdog: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        for feed in self.feeds.values():
            feed.start()
            log.info("camera %s started (source %r)", feed.name_tag,
                     feed.source)
        # web (waitress in a daemon thread)
        web = create_app(self.db, self.config, self.state)
        from waitress import serve

        self.web_thread = threading.Thread(
            target=serve,
            args=(web,),
            kwargs={"host": self.config.web.host,
                    "port": self.config.web.port},
            daemon=True,
            name="gate-web",
        )
        self.web_thread.start()
        log.info("web UI on http://%s:%d", self.config.web.host,
                 self.config.web.port)
        if self.with_display:
            self.display = StatusDisplay(
                self.db, self.state, self.config.display.width,
                self.config.display.height,
            )
            self.display.start()
            log.info("status display started")
        self.watchdog = threading.Thread(target=self._watchdog_loop,
                                         daemon=True, name="gate-watchdog")
        self.watchdog.start()

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        log.info("stopping...")
        for feed in self.feeds.values():
            feed.stop()
        if self.display is not None:
            self.display.stop()
        self.leds.close()
        try:
            self.db.close()
        except Exception:  # pragma: no cover
            pass
        log.info("stopped")

    def run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(1.0)

    # -- watchdog ----------------------------------------------------------

    def _restart_feed(self, side: str) -> None:
        cam = self.config.cameras[side]
        ctrl = self.controllers[side]
        feed = CameraFeed(side, cam.source, cam.width, cam.height, cam.fps,
                          ctrl.on_frame)
        self.feeds[side] = feed
        self.state.attach(side, feed)
        feed.start()
        log.warning("camera %s thread restarted", side)

    def _watchdog_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(WATCHDOG_S)
            for side, feed in list(self.feeds.items()):
                if not feed.is_alive():
                    log.error("camera %s thread dead; restarting", side)
                    try:
                        self._restart_feed(side)
                    except Exception:
                        log.exception("camera %s restart failed", side)
                elif feed.ok and time.monotonic() - feed.last_frame_at > 15.0:
                    # stream reports ok but frames stopped flowing: reopen
                    log.warning("camera %s stalled (no frames > 15 s); "
                                "restarting", side)
                    try:
                        self._restart_feed(side)
                    except Exception:
                        log.exception("camera %s restart failed", side)
            if not self.db.ping():
                log.error("database ping failed")
            # image retention housekeeping (bounded disk usage)
            try:
                self.storage.prune(self.config.storage.retention_days)
            except Exception:
                log.exception("image prune failed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VN Plate Gate")
    parser.add_argument("--config", default="/etc/gate/config.toml")
    parser.add_argument("--display", action="store_true",
                        help="enable the framebuffer status display")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        log.error("config not found: %s", args.config)
        return 1

    app = GateApp(config, with_display=args.display)
    app.start()

    def _sig(_signum, _frame):
        app.stop()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        app.run()
    finally:
        app.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
