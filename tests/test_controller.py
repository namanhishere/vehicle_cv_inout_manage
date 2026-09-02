"""Tests for gate/controller.py - the per-side burst pipeline.

Drives SideController.on_frame directly with synthetic composite frames
(motion + plate). Requires the trained detector + OCR models (Step 5).
"""

import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gate.capture import MotionTrigger
from gate.controller import SideController
from gate.db import GateDB
from gate.decision import DecisionEngine
from gate.leds import MockLedController
from gate.storage import Storage
from gate.vision.detector import YoloPlateDetector
from gate.vision.ocr import OcrEngine
from gate.vision.pipeline import PlateReader

from tests.conftest import scene_for

PLATE = "29A1-678.90"
UNREG = "29B7-111.11"

pytestmark = pytest.mark.skipif(
    not os.path.exists("models/plate_det.onnx"),
    reason="plate_det.onnx not present yet",
)


def split_plate(text):
    """'29A1-678.90' -> ('29-A1', '678.90') rendered line texts.

    Real plates carry the separator dash on the top line; without it the
    trailing digit sits at an odd spacing and OCR misreads 1 as l."""
    return f"{text[:2]}-{text[2:4]}", text[5:]


@pytest.fixture
def env(tmp_path):
    db = GateDB(str(tmp_path / "gate.db"))
    db.add_vehicle(PLATE)
    det = YoloPlateDetector("models/plate_det.onnx", conf=0.25, iou=0.45)
    reader = PlateReader(det, OcrEngine("models/ocr_rec.onnx"))
    engine = DecisionEngine(db, min_confidence=0.75, cooldown_s=10.0)
    leds = MockLedController(
        {"in_green": 17, "in_red": 27, "out_green": 22, "out_red": 23}
    )
    ctrl = SideController(
        "IN", camera=None, trigger=MotionTrigger(threshold=0.02, debounce_s=0.0),
        reader=reader, engine=engine, db=db, leds=leds,
        storage=Storage(str(tmp_path / "images")),
        burst_frames=4, min_frames_detected=2,
    )
    yield SimpleNamespace(db=db, ctrl=ctrl, leds=leds, engine=engine,
                          tmp=tmp_path)
    db.close()


def push_passage(env, text, seed=5, frames=8):
    """Feed frames that trigger motion, then hold the plate for a burst."""
    top, bottom = split_plate(text)
    # plate moving across the frame -> motion trigger
    for i in range(3):
        env.ctrl.on_frame(scene_for(top, bottom, seed=seed + i))
    base = scene_for(top, bottom, seed=seed + 10)
    for _ in range(frames):
        env.ctrl.on_frame(base)


def test_allow_in_records_event_state_and_led(env):
    push_passage(env, PLATE)
    events, total = env.db.list_events(page=1)
    assert total == 1
    ev = events[0]
    assert ev["direction"] == "IN" and ev["camera"] == "IN"
    assert ev["result"] == "ALLOW" and ev["reason"] == "ALLOW"
    assert ev["plate"] == PLATE
    assert ev["confidence"] >= 0.5
    assert env.db.lookup(PLATE)["inside"] == 1
    assert ("IN", "ALLOW") in [(s, o) for s, o, _t in env.leds.records]


def test_no_detection_no_event(env):
    # motion (moving white box) but no plate anywhere
    for x in range(0, 260, 20):
        fr = np.full((360, 640, 3), 128, dtype=np.uint8)
        fr[120:240, x:x + 60] = 255
        env.ctrl.on_frame(fr)
    assert env.db.list_events(page=1)[1] == 0
    assert env.leds.records == []


def test_unregistered_rejects(env):
    push_passage(env, UNREG)
    events, total = env.db.list_events(page=1)
    assert total == 1
    ev = events[0]
    assert ev["result"] == "REJECT"
    assert ev["reason"] == "UNREGISTERED"
    assert ev["plate"] == UNREG
    assert env.db.lookup(PLATE)["inside"] == 0
    assert ("IN", "REJECT") in [(s, o) for s, o, _t in env.leds.records]


def test_cooldown_suppresses_second_event(env):
    push_passage(env, PLATE)
    assert env.db.list_events(page=1)[1] == 1
    push_passage(env, PLATE, seed=30)  # another passage moments later
    assert env.db.list_events(page=1)[1] == 1  # cooldown: no second event


def test_low_confidence_rejects(env):
    # The synthetic-text OCR model reads clean renders all-or-nothing (it
    # either decodes confidently or not at all), so a "blurred fixture"
    # produces no read instead of a low-confidence one. Drive the same
    # controller path with an engine whose min_confidence sits above the
    # render confidences: valid plate text at conf < min -> LOW_CONF.
    import gate.decision as _dec

    strict = DecisionEngine(env.db, min_confidence=0.995, cooldown_s=10.0)
    ctrl = SideController(
        "IN", camera=None, trigger=MotionTrigger(threshold=0.02, debounce_s=0.0),
        reader=env.ctrl.reader, engine=strict, db=env.db,
        leds=env.leds, storage=env.ctrl.storage,
        burst_frames=4, min_frames_detected=2,
    )
    top, bottom = split_plate(PLATE)
    for i in range(3):
        ctrl.on_frame(scene_for(top, bottom, seed=9 + i))
    base = scene_for(top, bottom, seed=19)
    for _ in range(8):
        ctrl.on_frame(base)
    events, total = env.db.list_events(page=1)
    assert total >= 1
    # LOW_CONF is deliberately not cooldown-deduped (motion re-triggers can
    # re-evaluate): every event must be a LOW_CONF rejection
    assert all(e["result"] == "REJECT" and e["reason"] == "LOW_CONF"
               for e in events)
    assert events[0]["plate"] == PLATE
    assert env.db.lookup(PLATE)["inside"] == 0  # no state change
    assert ("IN", "LOW_CONF") in [(s, o) for s, o, _t in env.leds.records]


def test_out_while_outside_rejects(env):
    ctrl_out = SideController(
        "OUT", camera=None,
        trigger=MotionTrigger(threshold=0.02, debounce_s=0.0),
        reader=env.ctrl.reader, engine=env.engine, db=env.db,
        leds=env.leds, storage=env.ctrl.storage,
        burst_frames=4, min_frames_detected=2,
    )
    top, bottom = split_plate(PLATE)
    for i in range(3):  # moving plate -> motion trigger on the OUT side
        ctrl_out.on_frame(scene_for(top, bottom, seed=11 + i))
    base = scene_for(top, bottom, seed=21)
    for _ in range(8):
        ctrl_out.on_frame(base)
    events, total = env.db.list_events(page=1)
    assert total == 1
    assert events[0]["direction"] == "OUT"
    assert events[0]["result"] == "REJECT"
    assert events[0]["reason"] == "ALREADY_OUTSIDE"


def test_allow_saves_crop(env):
    push_passage(env, PLATE)
    events, _ = env.db.list_events(page=1)
    assert events and events[0]["crop"]
    crop_path = env.tmp / "images" / events[0]["crop"]
    assert crop_path.exists()
    assert crop_path.read_bytes()[:2] == b"\xff\xd8"  # JPEG magic
