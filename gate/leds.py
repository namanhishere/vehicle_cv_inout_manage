"""GPIO LED signaling (green/red per side) with a mock backend for tests.

Two backends behind one interface:

- ``GpioLedController`` - libgpiod v2 on /dev/gpiochip0. All chip access is
  wrapped: an LED failure must never break the gate logic (log and continue).
- ``MockLedController`` - records signals into an in-memory list.

Signaling semantics: ALLOW -> green solid for allow_s; REJECT -> red solid
for reject_s; LOW_CONF -> red blink (3 cycles of blink_s on/off).
Re-signaling the same side cancels the previous signal's timers.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

log = logging.getLogger("gate.leds")

try:
    import gpiod
except Exception:  # pragma: no cover - no libgpiod on dev workstations
    gpiod = None


class LedController:
    """Timer-driven per-side LED signaling.

    Subclasses implement ``_set(name, on)`` and ``_all_pins_off()`` where
    ``name`` is one of in_green/in_red/out_green/out_red.
    """

    _COLORS = {"IN": ("in_green", "in_red"), "OUT": ("out_green", "out_red")}
    _OUTCOMES = ("ALLOW", "REJECT", "LOW_CONF")

    def __init__(
        self,
        pins: dict,
        allow_s: float = 2.0,
        reject_s: float = 2.0,
        blink_s: float = 0.2,
    ):
        self._pins = {str(k): int(v) for k, v in pins.items()}
        self._allow_s = float(allow_s)
        self._reject_s = float(reject_s)
        self._blink_s = float(blink_s)
        self._lock = threading.RLock()
        self._pending: dict[str, list[threading.Timer]] = {}

    # -- subclass hooks ----------------------------------------------------

    def _set(self, name: str, on: bool) -> None:
        raise NotImplementedError

    def _all_pins_off(self) -> None:
        for name in self._pins:
            self._set(name, False)

    # -- signaling ---------------------------------------------------------

    def signal(self, side: str, outcome: str, confidence: float) -> None:
        """Signal one side. Idempotent: re-signaling the same side cancels
        the previous signal's timers before applying the new one."""
        if side not in self._COLORS or outcome not in self._OUTCOMES:
            return
        with self._lock:
            self._cancel(side)
            green, red = self._COLORS[side]
            if outcome == "ALLOW":
                self._set(green, True)
                self._schedule(side, self._allow_s, lambda: self._set(green, False))
            elif outcome == "REJECT":
                self._set(red, True)
                self._schedule(side, self._reject_s, lambda: self._set(red, False))
            else:  # LOW_CONF: red blink, 3 cycles of blink_s on/off
                for phase in range(6):  # on/off/on/off/on/off
                    on = phase % 2 == 0
                    self._schedule(
                        side, phase * self._blink_s,
                        lambda on=on: self._set(red, on),
                    )

    def all_off(self) -> None:
        with self._lock:
            for side in list(self._pending):
                self._cancel(side)
            self._all_pins_off()

    def close(self) -> None:
        self.all_off()

    # -- timer plumbing ----------------------------------------------------

    def _schedule(self, side: str, delay: float, fn: Callable[[], None]) -> None:
        timer = threading.Timer(max(0.0, delay), self._fire, args=(side, fn))
        self._pending.setdefault(side, []).append(timer)
        timer.daemon = True
        timer.start()

    def _fire(self, side: str, fn: Callable[[], None]) -> None:
        with self._lock:
            fn()
            timers = self._pending.get(side)
            if timers:
                timers.pop(0)

    def _cancel(self, side: str) -> None:
        for timer in self._pending.pop(side, []):
            timer.cancel()


class GpioLedController(LedController):
    """libgpiod v2 backend. Never raises into the caller: any chip error is
    logged and LEDs stay off."""

    def __init__(
        self,
        pins: dict,
        allow_s: float = 2.0,
        reject_s: float = 2.0,
        blink_s: float = 0.2,
        chip_path: str = "/dev/gpiochip0",
    ):
        super().__init__(pins, allow_s, reject_s, blink_s)
        self._chip = None
        self._req = None
        self._chip_path = chip_path
        self._open()

    def _open(self) -> None:
        if gpiod is None:
            log.warning("gpiod module unavailable; LEDs disabled")
            return
        try:
            chip = gpiod.Chip(self._chip_path)
            settings = {
                offset: gpiod.LineSettings(
                    direction=gpiod.line.Direction.OUTPUT
                )
                for offset in self._pins.values()
            }
            req = chip.request_lines(config=settings, consumer="gate-leds")
        except OSError as exc:
            log.warning("GPIO unavailable on %s: %s", self._chip_path, exc)
            return
        self._chip = chip
        self._req = req

    def _set(self, name: str, on: bool) -> None:
        if self._req is None:
            return
        offset = self._pins[name]
        try:
            self._req.set_value(
                offset,
                gpiod.line.Value.ACTIVE if on else gpiod.line.Value.INACTIVE,
            )
        except OSError as exc:
            log.warning("GPIO write failed on line %s: %s", offset, exc)

    def close(self) -> None:
        super().close()
        if self._chip is not None:
            try:
                self._chip.close()
            except OSError:
                pass


class MockLedController(LedController):
    """In-memory backend: ``records`` holds the current signal per side as
    ``(side, outcome, ts)``; re-signaling the same side replaces its entry
    (its prior signal is cancelled); ``all_off`` clears the list."""

    def __init__(
        self,
        pins: dict,
        allow_s: float = 2.0,
        reject_s: float = 2.0,
        blink_s: float = 0.2,
    ):
        super().__init__(pins, allow_s, reject_s, blink_s)
        self.records: list[tuple[str, str, float]] = []

    def signal(self, side: str, outcome: str, confidence: float) -> None:
        if side not in self._COLORS or outcome not in self._OUTCOMES:
            return
        with self._lock:
            self._cancel(side)
            self.records = [r for r in self.records if r[0] != side]
            self.records.append((side, outcome, time.time()))

    def all_off(self) -> None:
        with self._lock:
            for side in list(self._pending):
                self._cancel(side)
            self.records = []

    def _set(self, name: str, on: bool) -> None:  # unused by the mock
        pass
