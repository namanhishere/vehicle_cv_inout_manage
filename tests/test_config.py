"""Tests for gate/config.py - TOML loading with defaults."""

from gate.config import load_config

DEFAULTS = """
[cameras.in]
source = "1"

[vision]
min_confidence = 0.9

[unknown_section]
whatever = 1

[web]
unknown_key = 5
"""


def test_defaults_and_overrides(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(DEFAULTS)
    cfg = load_config(str(p))
    # defaults where unspecified
    assert cfg.cameras["IN"].width == 640
    assert cfg.cameras["OUT"].source == "0"
    assert cfg.vision.burst_frames == 5
    assert cfg.decision.cooldown_s == 10.0
    assert cfg.leds.in_green == 17
    assert cfg.storage.db_path == "/var/lib/gate/gate.db"
    assert cfg.web.host == "0.0.0.0"
    assert cfg.display.enabled is False
    # overrides applied
    assert cfg.cameras["IN"].source == "1"
    assert cfg.vision.min_confidence == 0.9
    # unknown keys/sections ignored
    assert cfg.web.port == 8080


def test_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.toml"))
