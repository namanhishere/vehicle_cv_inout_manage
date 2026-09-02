"""Tests for gate/leds.py - LED signaling (mock backend, gpio degrade)."""

from gate.leds import GpioLedController, MockLedController

PINS = {"in_green": 17, "in_red": 27, "out_green": 22, "out_red": 23}


def _mock():
    return MockLedController(PINS, allow_s=2.0, reject_s=2.0, blink_s=0.2)


def pairs(records):
    return sorted((side, outcome) for side, outcome, _ts in records)


def test_mock_records_outcomes_per_side():
    leds = _mock()
    leds.signal("IN", "ALLOW", 0.9)
    assert pairs(leds.records) == [("IN", "ALLOW")]
    leds.signal("OUT", "REJECT", 0.8)
    leds.signal("IN", "LOW_CONF", 0.6)
    assert pairs(leds.records) == [("IN", "LOW_CONF"), ("OUT", "REJECT")]


def test_resignal_same_side_cancels_prior():
    leds = _mock()
    leds.signal("IN", "ALLOW", 0.9)
    leds.signal("IN", "ALLOW", 0.95)  # re-trigger replaces, does not append
    assert len(leds.records) == 1
    assert leds.records[0][:2] == ("IN", "ALLOW")
    leds.signal("IN", "REJECT", 0.8)  # outcome flip also replaces
    assert len(leds.records) == 1
    assert leds.records[0][:2] == ("IN", "REJECT")


def test_all_off_clears():
    leds = _mock()
    leds.signal("IN", "ALLOW", 0.9)
    leds.signal("OUT", "LOW_CONF", 0.6)
    leds.all_off()
    assert leds.records == []


def test_invalid_side_or_outcome_ignored():
    leds = _mock()
    leds.signal("XX", "ALLOW", 0.9)
    leds.signal("IN", "BOGUS", 0.9)
    assert leds.records == []


def test_gpio_fake_chip_path_fails_gracefully():
    leds = GpioLedController(
        PINS, allow_s=1.0, reject_s=1.0, blink_s=0.1,
        chip_path="/dev/gpiochip-does-not-exist",
    )
    # construction and signaling must not raise
    leds.signal("IN", "ALLOW", 0.9)
    leds.signal("OUT", "LOW_CONF", 0.5)
    leds.all_off()
    leds.close()
    assert leds._req is None  # degraded, not fatal


def test_mock_timestamps_monotonic():
    leds = _mock()
    leds.signal("IN", "ALLOW", 0.9)
    t1 = leds.records[0][2]
    leds.signal("IN", "ALLOW", 0.9)
    t2 = leds.records[0][2]
    assert t2 >= t1
