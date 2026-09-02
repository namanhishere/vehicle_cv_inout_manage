"""Gate configuration: TOML file -> typed Config dataclass.

Unknown keys are ignored; missing keys fall back to the defaults below
(exact). The template written by the installer (not by the app) is
documented in docs/DEPLOY.md.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    source: str = "0"  # V4L2 index, or "rtsp://..." URL
    width: int = 640
    height: int = 360
    fps: int = 10


@dataclass
class VisionConfig:
    burst_frames: int = 5
    min_frames_detected: int = 2
    min_confidence: float = 0.75
    model_dir: str = "/opt/gate/models"


@dataclass
class DecisionConfig:
    cooldown_s: float = 10.0


@dataclass
class LedsConfig:
    in_green: int = 17
    in_red: int = 27
    out_green: int = 22
    out_red: int = 23
    allow_s: float = 2.0
    reject_s: float = 2.0
    blink_s: float = 0.2


@dataclass
class StorageConfig:
    db_path: str = "/var/lib/gate/gate.db"
    images_dir: str = "/var/lib/gate/images"
    retention_days: int = 7


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    secret_file: str = "/var/lib/gate/secret"
    password_file: str = "/var/lib/gate/admin.hash"


@dataclass
class DisplayConfig:
    enabled: bool = False  # true when run with --display
    width: int = 640
    height: int = 480


@dataclass
class Config:
    cameras: dict = field(default_factory=lambda: {
        "IN": CameraConfig(),
        "OUT": CameraConfig(),
    })
    vision: VisionConfig = field(default_factory=VisionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    leds: LedsConfig = field(default_factory=LedsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    web: WebConfig = field(default_factory=WebConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


_SECTIONS = {
    "vision": VisionConfig,
    "decision": DecisionConfig,
    "leds": LedsConfig,
    "storage": StorageConfig,
    "web": WebConfig,
    "display": DisplayConfig,
}

_KEYS = {
    "cameras": {"in": CameraConfig, "out": CameraConfig},
}


def _apply(section: dict, cls) -> object:
    """Build cls(), overriding attributes present in the TOML section."""
    known = {k for k in cls.__dataclass_fields__}
    cfg = cls()
    for key, value in section.items():
        if key in known and value is not None:
            setattr(cfg, key, value)
    return cfg


def load_config(path: str) -> Config:
    """Read a TOML config file into a Config (unknown keys ignored)."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    cfg = Config()
    for name, cls in _SECTIONS.items():
        if name in data and isinstance(data[name], dict):
            setattr(cfg, name, _apply(data[name], cls))
    cams = data.get("cameras")
    if isinstance(cams, dict):
        for side, key in (("IN", "in"), ("OUT", "out")):
            sec = cams.get(key)
            if isinstance(sec, dict):
                cfg.cameras[side] = _apply(sec, CameraConfig)
    return cfg
