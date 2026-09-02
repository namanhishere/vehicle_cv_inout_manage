"""SQLite storage layer for the gate.

Single shared connection (WAL mode), one RLock serializing every operation.
``GateDB`` is the ONLY SQLite surface - no raw connection is ever exposed
(web/controller layers never touch sqlite directly).

All plate inputs must be canonical strings validated by ``plate.normalize``
BEFORE reaching ``GateDB``; add_vehicle/lookup re-check and raise ValueError
on anything non-canonical.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from typing import Any

from gate.plate import normalize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
  plate TEXT PRIMARY KEY,            -- canonical
  registered INTEGER NOT NULL DEFAULT 1,
  inside INTEGER NOT NULL DEFAULT 0,
  note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,                  -- ISO-8601 local time
  plate TEXT NOT NULL,               -- canonical, '' when invalid
  raw TEXT NOT NULL,                 -- raw OCR string
  direction TEXT NOT NULL CHECK (direction IN ('IN','OUT')),
  result TEXT NOT NULL CHECK (result IN ('ALLOW','REJECT')),
  reason TEXT NOT NULL,              -- ALLOW|UNREGISTERED|INVALID_FORMAT|ALREADY_INSIDE|ALREADY_OUTSIDE|LOW_CONF
  confidence REAL,
  camera TEXT NOT NULL,              -- 'IN' | 'OUT'
  crop TEXT                          -- relative path under images_dir, or NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
"""


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _canonical_or_raise(plate: str) -> None:
    parsed = normalize(plate)
    if parsed is None or parsed.canonical != plate:
        raise ValueError(f"non-canonical plate: {plate!r}")


class GateDB:
    """Thread-safe SQLite store for vehicles and events."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA journal_mode = WAL")
        self.init_schema()

    def init_schema(self) -> None:
        """Apply the v1 schema; idempotent."""
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA user_version = 1")
            self._conn.commit()

    # -- vehicles ----------------------------------------------------------

    def add_vehicle(self, plate: str, note: str = "") -> None:
        """Insert a registered vehicle. Raises ValueError on duplicate or
        non-canonical input (caller must validate via plate.normalize first)."""
        _canonical_or_raise(plate)
        now = _now_iso()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO vehicles (plate, registered, inside, note,"
                    " created_at, updated_at) VALUES (?, 1, 0, ?, ?, ?)",
                    (plate, note, now, now),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"duplicate vehicle: {plate!r}") from exc

    def remove_vehicle(self, plate: str) -> None:
        _canonical_or_raise(plate)
        with self._lock:
            self._conn.execute("DELETE FROM vehicles WHERE plate = ?", (plate,))
            self._conn.commit()

    def set_registered(self, plate: str, registered: bool) -> None:
        _canonical_or_raise(plate)
        with self._lock:
            self._conn.execute(
                "UPDATE vehicles SET registered = ?, updated_at = ?"
                " WHERE plate = ?",
                (1 if registered else 0, _now_iso(), plate),
            )
            self._conn.commit()

    def set_inside(self, plate: str, inside: bool) -> None:
        _canonical_or_raise(plate)
        with self._lock:
            self._conn.execute(
                "UPDATE vehicles SET inside = ?, updated_at = ?"
                " WHERE plate = ?",
                (1 if inside else 0, _now_iso(), plate),
            )
            self._conn.commit()

    def lookup(self, plate: str) -> sqlite3.Row | None:
        _canonical_or_raise(plate)
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM vehicles WHERE plate = ?", (plate,)
            )
            return cur.fetchone()

    def inside_count(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM vehicles WHERE inside = 1"
            )
            return int(cur.fetchone()["n"])

    def list_vehicles(self) -> list[Any]:
        """All vehicle rows, ordered by plate."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM vehicles ORDER BY plate"
            ).fetchall()
            return list(rows)

    def ping(self) -> bool:
        """Live SELECT 1 - used by the watchdog and the web dashboard."""
        try:
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    # -- events ------------------------------------------------------------

    def record_event(
        self,
        *,
        ts: str,
        plate: str,
        raw: str,
        direction: str,
        result: str,
        reason: str,
        confidence: float | None,
        camera: str,
        crop: str | None,
    ) -> int:
        """Insert one event row; returns the new event id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (ts, plate, raw, direction, result,"
                " reason, confidence, camera, crop)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, plate, raw, direction, result, reason,
                 confidence, camera, crop),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_crop(self, event_id: int, crop_rel: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET crop = ? WHERE id = ?", (crop_rel, event_id)
            )
            self._conn.commit()

    def last_event(self) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM events ORDER BY ts DESC, id DESC LIMIT 1"
            )
            return cur.fetchone()

    def list_events(
        self, page: int, per_page: int = 25
    ) -> tuple[list[Any], int]:
        """(rows, total) - newest first, paginated."""
        page = max(1, int(page))
        per_page = max(1, int(per_page))
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM events"
            )
            total = int(cur.fetchone()["n"])
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY ts DESC, id DESC"
                " LIMIT ? OFFSET ?",
                (per_page, (page - 1) * per_page),
            ).fetchall()
            return list(rows), total

    def close(self) -> None:
        with self._lock:
            self._conn.close()
