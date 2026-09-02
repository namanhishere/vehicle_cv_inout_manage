"""Gate command-line tool.

Subcommands:
  init-db --config PATH        create/upgrade the database schema
  passwd --config PATH         set the admin password (scrypt hash file)
  add-vehicle PLATE [--note N] [--unregistered] --config PATH
  remove-vehicle PLATE --config PATH
  simulate --video PATH --side IN|OUT --config PATH   full pipeline over a
                               video file (records events into the DB)

Hash format: scrypt$<n>$<r>$<p>$<salt_hex>$<hash_hex>, n=2**14 r=8 p=1.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import sys

import cv2

from gate.capture import MotionTrigger
from gate.config import load_config
from gate.controller import SideController
from gate.db import GateDB
from gate.decision import DecisionEngine
from gate.leds import MockLedController
from gate.plate import normalize
from gate.storage import Storage
from gate.vision.detector import YoloPlateDetector
from gate.vision.ocr import OcrEngine
from gate.vision.pipeline import PlateReader

DEFAULT_CONFIG = "/etc/gate/config.toml"


def scrypt_hash(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt$16384$8$1${salt.hex()}${digest.hex()}"


def verify_scrypt(password: str, stored: str) -> bool:
    parts = stored.strip().split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return False
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt = bytes.fromhex(parts[4])
        expected = bytes.fromhex(parts[5])
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p)
    return hmac.compare_digest(digest, expected)


# -- subcommands ------------------------------------------------------------

def cmd_init_db(args) -> int:
    cfg = load_config(args.config)
    db = GateDB(cfg.storage.db_path)
    db.close()
    print(f"database ready at {cfg.storage.db_path}")
    return 0


def cmd_passwd(args) -> int:
    cfg = load_config(args.config)
    pw = getpass_getpass("New admin password: ")
    if not pw:
        print("password must not be empty", file=sys.stderr)
        return 1
    if pw != getpass_getpass("Repeat admin password: "):
        print("passwords do not match", file=sys.stderr)
        return 1
    path = cfg.web.password_file
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(scrypt_hash(pw) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    print(f"password hash written to {path}")
    return 0


def _require_plate(raw: str) -> str:
    parsed = normalize(raw)
    if parsed is None:
        print(f"invalid plate format: {raw!r}", file=sys.stderr)
        sys.exit(2)
    return parsed.canonical


def cmd_add_vehicle(args) -> int:
    cfg = load_config(args.config)
    canonical = _require_plate(args.plate)
    db = GateDB(cfg.storage.db_path)
    try:
        db.add_vehicle(canonical, note=args.note or "")
        if args.unregistered:
            db.set_registered(canonical, False)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()
    state = "unregistered" if args.unregistered else "registered"
    print(f"{canonical} {state}")
    return 0


def cmd_remove_vehicle(args) -> int:
    cfg = load_config(args.config)
    canonical = _require_plate(args.plate)
    db = GateDB(cfg.storage.db_path)
    try:
        db.remove_vehicle(canonical)
    finally:
        db.close()
    print(f"{canonical} removed")
    return 0


def cmd_simulate(args) -> int:
    """Run the whole per-side pipeline over a video file."""
    cfg = load_config(args.config)
    db = GateDB(cfg.storage.db_path)
    try:
        model_dir = cfg.vision.model_dir
        reader = PlateReader(
            YoloPlateDetector(os.path.join(model_dir, "plate_det.onnx")),
            OcrEngine(os.path.join(model_dir, "ocr_rec.onnx")),
        )
        engine = DecisionEngine(
            db, cfg.vision.min_confidence, cfg.decision.cooldown_s
        )
        pins = {
            "in_green": cfg.leds.in_green, "in_red": cfg.leds.in_red,
            "out_green": cfg.leds.out_green, "out_red": cfg.leds.out_red,
        }
        leds = MockLedController(
            pins, cfg.leds.allow_s, cfg.leds.reject_s, cfg.leds.blink_s
        )
        ctrl = SideController(
            args.side, camera=None, trigger=MotionTrigger(),
            reader=reader, engine=engine, db=db, leds=leds,
            storage=Storage(cfg.storage.images_dir),
            burst_frames=cfg.vision.burst_frames,
            min_frames_detected=cfg.vision.min_frames_detected,
        )
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"cannot open video: {args.video}", file=sys.stderr)
            return 1
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            ctrl.on_frame(frame)
        cap.release()
        events, total = db.list_events(page=1, per_page=100)
        print(f"simulated {frames} frames from {args.video} "
              f"(side {args.side})")
        for ev in reversed(events):
            print(
                f"  {ev['ts']} {ev['direction']} {ev['result']} "
                f"{ev['reason']} {ev['plate'] or '(none)'} "
                f"conf={ev['confidence']:.2f}"
            )
        if leds.records:
            print("  leds:", [(s, o) for s, o, _t in leds.records])
        return 0
    finally:
        db.close()


def getpass_getpass(prompt: str) -> str:
    import getpass

    return getpass.getpass(prompt)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gate.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="create/upgrade the schema")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("passwd", help="set the admin password")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.set_defaults(func=cmd_passwd)

    p = sub.add_parser("add-vehicle", help="register a plate")
    p.add_argument("plate")
    p.add_argument("--note", default="")
    p.add_argument("--unregistered", action="store_true")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.set_defaults(func=cmd_add_vehicle)

    p = sub.add_parser("remove-vehicle", help="unregister a plate")
    p.add_argument("plate")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.set_defaults(func=cmd_remove_vehicle)

    p = sub.add_parser("simulate", help="run the pipeline over a video")
    p.add_argument("--video", required=True)
    p.add_argument("--side", choices=("IN", "OUT"), required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
