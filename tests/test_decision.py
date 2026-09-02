"""Tests for gate/decision.py - access-control state machine."""

from types import SimpleNamespace

import pytest

from gate.db import GateDB
from gate.decision import DecisionEngine, Reason, Result

P1 = "29A1-678.90"  # LEGACY5
P2 = "29HA-002.33"  # NEW
MD_P = "29MD-002.12"  # NEW bare-MD canonical (latin D)
MĐ_P = "29MĐ-002.12"  # ELECTRIC canonical


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture
def env(tmp_path):
    db = GateDB(str(tmp_path / "d.db"))
    clock = Clock()
    eng = DecisionEngine(db, min_confidence=0.75, cooldown_s=10.0, now=clock)
    yield SimpleNamespace(db=db, eng=eng, clock=clock)
    db.close()


def classify(env, raw=P1, direction="IN", conf=0.9):
    return env.eng.classify(
        raw=raw, direction=direction, confidence=conf, camera=direction
    )


def test_allow_in_then_out_roundtrip(env):
    env.db.add_vehicle(P1)
    d = classify(env, direction="IN")
    assert (d.result, d.reason, d.plate) == (Result.ALLOW, Reason.ALLOW, P1)
    assert not d.duplicate
    env.eng.apply(d)
    assert env.db.lookup(P1)["inside"] == 1
    # OUT while inside -> ALLOW, state flips back
    d = classify(env, direction="OUT")
    assert (d.result, d.reason) == (Result.ALLOW, Reason.ALLOW)
    env.eng.apply(d)
    assert env.db.lookup(P1)["inside"] == 0


def test_already_inside_rejects(env):
    env.db.add_vehicle(P1)
    env.db.set_inside(P1, True)
    d = classify(env, direction="IN")
    assert (d.result, d.reason) == (Result.REJECT, Reason.ALREADY_INSIDE)
    env.eng.apply(d)  # no-op for rejects
    assert env.db.lookup(P1)["inside"] == 1


def test_already_outside_rejects(env):
    env.db.add_vehicle(P1)
    d = classify(env, direction="OUT")
    assert (d.result, d.reason) == (Result.REJECT, Reason.ALREADY_OUTSIDE)
    env.eng.apply(d)
    assert env.db.lookup(P1)["inside"] == 0


def test_unregistered_rejects(env):
    d = classify(env)
    assert (d.result, d.reason, d.plate) == (
        Result.REJECT, Reason.UNREGISTERED, P1,
    )
    assert d.confidence == 0.9


def test_registered_toggle_off_rejects(env):
    env.db.add_vehicle(P1)
    env.db.set_registered(P1, False)
    d = classify(env)
    assert (d.result, d.reason) == (Result.REJECT, Reason.UNREGISTERED)


def test_invalid_format_precedes_everything(env):
    for conf in (0.99, 0.1):
        d = classify(env, raw="not a plate 123!", conf=conf)
        assert (d.result, d.reason, d.plate) == (
            Result.REJECT, Reason.INVALID_FORMAT, "",
        )


def test_low_conf_precedes_registry_and_state(env):
    env.db.add_vehicle(P1)
    env.db.set_inside(P1, True)
    d = classify(env, conf=0.7)
    assert (d.result, d.reason, d.plate) == (
        Result.REJECT, Reason.LOW_CONF, P1,
    )
    d = classify(env, raw="unregistered-ish", conf=0.7)
    # INVALID_FORMAT wins over LOW_CONF when the raw is garbage
    assert (d.result, d.reason) == (Result.REJECT, Reason.INVALID_FORMAT)


@pytest.mark.parametrize(
    "raw,plate",
    [("29-HA 002.33", P2), ("29f1-1234", "29F1-1234")],
)
def test_raw_forms_normalize_then_allow(env, raw, plate):
    env.db.add_vehicle(plate)
    d = classify(env, raw=raw)
    assert (d.result, d.reason, d.plate) == (Result.ALLOW, Reason.ALLOW, plate)
    env.eng.apply(d)
    assert env.db.lookup(plate)["inside"] == 1


# -- cooldown --------------------------------------------------------------


def test_cooldown_duplicate_suppressed(env):
    env.db.add_vehicle(P1)
    d1 = classify(env)
    assert not d1.duplicate
    d2 = classify(env)
    assert d2.duplicate
    assert (d2.result, d2.reason, d2.plate) == (
        Result.ALLOW, Reason.ALLOW, P1,
    )
    d3 = classify(env)
    assert d3.duplicate


def test_cooldown_per_plate_and_direction(env):
    env.db.add_vehicle(P1)
    env.db.add_vehicle(P2)
    env.eng.apply(classify(env))  # P1 IN allowed -> inside=1
    # different direction is a distinct transition, not suppressed
    d_out = classify(env, direction="OUT")
    assert not d_out.duplicate
    assert d_out.reason is Reason.ALLOW
    env.eng.apply(d_out)  # back outside
    # different plate is not suppressed
    d_other = classify(env, raw=P2)
    assert not d_other.duplicate
    # original pair suppressed again
    assert classify(env).duplicate


def test_cooldown_expires(env):
    env.db.add_vehicle(P1)
    classify(env)
    assert classify(env).duplicate
    env.clock.advance(10.0)  # exactly cooldown_s: window closed
    d = classify(env)
    assert not d.duplicate
    assert d.reason is Reason.ALLOW  # apply() never called: state unchanged
    env.clock.advance(0.1)
    assert classify(env).duplicate  # fresh window from the last decision


def test_low_conf_and_invalid_not_cooldowned(env):
    env.db.add_vehicle(P1)
    d1 = classify(env, conf=0.7)
    d2 = classify(env, conf=0.7)
    assert not d1.duplicate and not d2.duplicate
    r1 = classify(env, raw="garbage")
    r2 = classify(env, raw="garbage")
    assert not r1.duplicate and not r2.duplicate


def test_rejects_are_cooldowned(env):
    d1 = classify(env)  # UNREGISTERED
    d2 = classify(env)
    assert (d1.result, d1.reason) == (Result.REJECT, Reason.UNREGISTERED)
    assert d2.duplicate


# -- MD / MĐ disambiguation -------------------------------------------------


def test_md_registered_only_uses_md(env):
    env.db.add_vehicle(MD_P)
    d = classify(env, raw="29-MD 002.12")
    assert (d.result, d.reason, d.plate) == (
        Result.ALLOW, Reason.ALLOW, MD_P,
    )
    env.eng.apply(d)
    assert env.db.lookup(MD_P)["inside"] == 1


def test_md_registered_only_uses_electric_form(env):
    env.db.add_vehicle(MĐ_P)
    d = classify(env, raw="29-MD 002.12")
    assert (d.result, d.reason, d.plate) == (
        Result.ALLOW, Reason.ALLOW, MĐ_P,
    )
    env.eng.apply(d)
    assert env.db.lookup(MĐ_P)["inside"] == 1


def test_md_both_registered_ambiguous(env):
    env.db.add_vehicle(MD_P)
    env.db.add_vehicle(MĐ_P)
    d = classify(env, raw="29-MD 002.12")
    assert (d.result, d.reason) == (Result.REJECT, Reason.LOW_CONF)


def test_md_neither_registered_treated_as_electric(env):
    d = classify(env, raw="29-MD 002.12")
    assert (d.result, d.reason, d.plate) == (
        Result.REJECT, Reason.UNREGISTERED, MĐ_P,
    )


def test_electric_kind_raw_skips_disambiguation(env):
    env.db.add_vehicle(MĐ_P)
    d = classify(env, raw="29MĐ 002.12")  # kind ELECTRIC directly
    assert (d.result, d.plate) == (Result.ALLOW, MĐ_P)


def test_md_disambiguation_duplicate_keeps_resolved_plate(env):
    env.db.add_vehicle(MĐ_P)
    d1 = classify(env, raw="29-MD 002.12")
    env.eng.apply(d1)
    d2 = classify(env, raw="29-MD 002.12")
    assert d2.duplicate
    assert d2.plate == MĐ_P
