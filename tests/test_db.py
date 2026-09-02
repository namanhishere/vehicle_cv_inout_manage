"""Tests for gate/db.py and gate/storage.py."""

import re
import sqlite3
from datetime import date, timedelta

import pytest

from gate.db import GateDB
from gate.storage import crop_path, prune, save_crop

PLATE = "29A1-678.90"
PLATE2 = "29HA-002.33"


@pytest.fixture
def db(tmp_path):
    d = GateDB(str(tmp_path / "gate.db"))
    yield d
    d.close()


def test_schema_version_applied(tmp_path):
    path = str(tmp_path / "gate.db")
    GateDB(path).close()
    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {
            r[0]
            for r in raw.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"vehicles", "events"} <= tables
    finally:
        raw.close()


def test_crud_roundtrip(db):
    db.add_vehicle(PLATE, note="resident")
    row = db.lookup(PLATE)
    assert row is not None
    assert row["registered"] == 1
    assert row["inside"] == 0
    assert row["note"] == "resident"
    db.set_registered(PLATE, False)
    assert db.lookup(PLATE)["registered"] == 0
    db.set_inside(PLATE, True)
    assert db.lookup(PLATE)["inside"] == 1
    db.remove_vehicle(PLATE)
    assert db.lookup(PLATE) is None


def test_duplicate_insert_raises(db):
    db.add_vehicle(PLATE)
    with pytest.raises(ValueError):
        db.add_vehicle(PLATE)


@pytest.mark.parametrize(
    "bad",
    ["", "hello", "29AB123.45", "29AB-123.45 ",
     "29A1-678.90'; DROP TABLE vehicles;--"],
)
def test_non_canonical_input_raises(db, bad):
    with pytest.raises(ValueError):
        db.add_vehicle(bad)
    with pytest.raises(ValueError):
        db.lookup(bad)


def test_parameterized_queries_never_interpolate(db):
    raw = "'; DROP TABLE vehicles;--"
    eid = db.record_event(
        ts="2026-01-01T00:00:00", plate="", raw=raw, direction="IN",
        result="REJECT", reason="INVALID_FORMAT", confidence=0.9,
        camera="IN", crop=None,
    )
    ev = db.last_event()
    assert ev is not None and ev["raw"] == raw and ev["id"] == eid
    # vehicles table still intact
    db.add_vehicle(PLATE)
    assert db.lookup(PLATE) is not None


def test_inside_count(db):
    db.add_vehicle(PLATE)
    db.add_vehicle(PLATE2)
    assert db.inside_count() == 0
    db.set_inside(PLATE, True)
    assert db.inside_count() == 1
    db.set_inside(PLATE, False)
    db.set_inside(PLATE2, True)
    assert db.inside_count() == 1


def _mk_event(db, ts, plate=PLATE, result="ALLOW", reason="ALLOW"):
    return db.record_event(
        ts=ts, plate=plate, raw=plate, direction="IN", result=result,
        reason=reason, confidence=0.9, camera="IN", crop=None,
    )


def test_list_events_pagination_order(db):
    for i in range(3):
        _mk_event(db, f"2026-01-0{i + 1}T00:00:00")
    rows, total = db.list_events(page=1, per_page=2)
    assert total == 3
    assert len(rows) == 2
    assert [r["ts"] for r in rows] == [
        "2026-01-03T00:00:00", "2026-01-02T00:00:00",
    ]
    rows2, _ = db.list_events(page=2, per_page=2)
    assert [r["ts"] for r in rows2] == ["2026-01-01T00:00:00"]


def test_last_event_ordering(db):
    _mk_event(db, "2026-01-01T00:00:00")
    _mk_event(db, "2026-01-02T00:00:00")
    last = db.last_event()
    assert last["ts"] == "2026-01-02T00:00:00"
    assert db.last_event()["plate"] == PLATE


def test_record_event_columns(db):
    eid = db.record_event(
        ts="2026-01-01T00:00:00", plate=PLATE, raw="29-A1 678.90",
        direction="OUT", result="ALLOW", reason="ALLOW", confidence=0.87,
        camera="OUT", crop="2026/01/01/00000001_plate.jpg",
    )
    assert eid == 1
    assert db.last_event()["direction"] == "OUT"
    assert db.last_event()["camera"] == "OUT"


def test_crop_path_shape():
    rel = crop_path("/var/lib/gate/images", 42)
    assert re.fullmatch(r"\d{4}/\d{2}/\d{2}/00000042_plate\.jpg", rel)
    # local date prefix
    today = date.today()
    assert rel.startswith(f"{today:%Y}/{today:%m}/{today:%d}/")


def test_save_crop_roundtrip(tmp_path):
    rel = save_crop(str(tmp_path), 7, b"\xff\xd8fakejpeg")
    assert rel.endswith("_plate.jpg")
    full = tmp_path / rel
    assert full.read_bytes() == b"\xff\xd8fakejpeg"


def test_prune_removes_old_keeps_new(tmp_path):
    images = tmp_path / "images"
    today = date.today()

    def mk(days_ago: int):
        d = today - timedelta(days=days_ago)
        p = images / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
        p.mkdir(parents=True)
        (p / "x.jpg").write_bytes(b"j")

    mk(30)
    mk(8)
    mk(7)  # boundary: kept (age == retention)
    mk(1)
    prune(str(images), retention_days=7)
    remaining = sorted(
        str(p.relative_to(images))
        for p in images.rglob("*")
        if p.is_dir() and len(str(p.relative_to(images)).split("/")) == 3
    )
    assert len(remaining) == 2  # 7-day and 1-day dirs survive
    for rel in remaining:
        assert rel.startswith(f"{today - timedelta(days=7):%Y/%m/%d}") or \
            rel.startswith(f"{today - timedelta(days=1):%Y/%m/%d}")
