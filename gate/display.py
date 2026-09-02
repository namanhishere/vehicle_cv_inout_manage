"""Framebuffer status display (pygame/SDL) for the "with desktop" variant.

Headless variant runs identical code with the display simply not started.
Rendering: black background; top bar GATE SYSTEM + ONLINE/camera-FAIL line;
vehicles-inside count (huge digits); last event plate (huge), direction
arrow, ALLOW (green box) / REJECT (red box) result, timestamp. ALLOW/REJECT
flash the result box color for the LED duration (2 s).

On the Pi, SDL_VIDEODRIVER=fbcon SDL_FBDEV=/dev/fb0; tests run with
SDL_VIDEODRIVER=dummy. Frames are double-buffered; rendered surfaces are
cached (never re-rendered per frame unless the value changed).
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # safe default off-Pi

import pygame

FLASH_S = 2.0
FPS = 10


class StatusDisplay(threading.Thread):
    def __init__(self, db, state, width: int = 640, height: int = 480):
        super().__init__(daemon=True, name="status-display")
        self.db = db
        self.state = state  # RuntimeState: camera_ok(side)
        self.width = width
        self.height = height
        self.screen = None
        self._stop_event = threading.Event()
        self._text_cache: dict[tuple, pygame.Surface] = {}
        self._flash_until = 0.0
        self._flash_color = None
        self._last_event_key = None

    # -- lifecycle ---------------------------------------------------------

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5.0)

    def run(self) -> None:
        pygame.init()
        pygame.display.set_mode((self.width, self.height))
        self.screen = pygame.display.get_surface()
        self._fonts = {
            "small": pygame.font.SysFont("dejavusansmono", 22),
            "big": pygame.font.SysFont("dejavusansmono", 44),
            "huge": pygame.font.SysFont("dejavusansmono", 72),
        }
        frame_s = 1.0 / FPS
        while not self._stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_event.set()
            self._draw()
            pygame.display.flip()
            self._stop_event.wait(frame_s)
        pygame.quit()

    # -- text cache --------------------------------------------------------

    def _text(self, font_key: str, text: str, color):
        key = (font_key, text, color)
        surf = self._text_cache.get(key)
        if surf is None:
            surf = self._fonts[font_key].render(text, True, color)
            if len(self._text_cache) > 48:  # bounded: never grow unbounded
                self._text_cache.clear()
            self._text_cache[key] = surf
        return surf

    # -- drawing -----------------------------------------------------------

    def _draw(self) -> None:
        if self.screen is None:
            return
        screen = self.screen
        screen.fill((0, 0, 0))
        w, h = self.width, self.height

        cams_ok = all(
            self.state.camera_ok(side) for side in ("IN", "OUT")
        )
        status = "ONLINE" if cams_ok else "CAMERA FAIL"
        status_color = (80, 220, 110) if cams_ok else (240, 90, 90)

        # top bar
        screen.blit(self._text("small", "GATE SYSTEM", (200, 200, 200)), (16, 12))
        screen.blit(self._text("small", status, status_color), (w - 200, 12))

        # middle: vehicles inside
        count = str(self.db.inside_count())
        screen.blit(self._text("big", "INSIDE", (150, 150, 150)),
                    (16, h // 2 - 70))
        screen.blit(self._text("huge", count, (255, 255, 255)), (16, h // 2 - 26))

        # bottom: last event
        event = self.db.last_event()
        now = time.monotonic()
        if event is None:
            screen.blit(self._text("big", "NO EVENTS", (120, 120, 120)),
                        (16, h - 110))
            return
        key = (event["ts"], event["plate"], event["result"])
        if key != self._last_event_key:
            self._last_event_key = key
            self._flash_until = now + FLASH_S
            self._flash_color = (80, 220, 110) if event["result"] == "ALLOW" \
                else (240, 90, 90)
        flashing = self._flash_color is not None and now < self._flash_until
        box_color = self._flash_color if flashing else (60, 60, 60)

        plate = event["plate"] or "(unreadable)"
        arrow = "\u2191 IN" if event["direction"] == "IN" else "\u2193 OUT"
        y = h - 108
        pygame.draw.rect(screen, box_color, (16, y, 200, 64), border_radius=8)
        screen.blit(
            self._text("big", event["result"], (0, 0, 0)), (28, y + 12)
        )
        screen.blit(self._text("big", plate, (255, 255, 255)), (240, y))
        screen.blit(self._text("big", arrow, (150, 200, 255)), (w - 220, y))
        screen.blit(
            self._text("small", event["ts"][11:19] if len(event["ts"]) >= 19
                       else event["ts"], (170, 170, 170)),
            (w - 220, y + 34),
        )
