# VN Plate Gate - Deployment (Raspberry Pi)

Target: **Raspberry Pi OS Bookworm 64-bit (arm64)** on a Raspberry Pi
Zero 2 W (512 MB RAM). The runtime makes **no network calls anywhere**
(fully offline): detection + OCR run locally via onnxruntime (CPU).

## 1. Hardware

- 2 cameras (IN, OUT): USB V4L2 or RTSP network cameras.
  - Find V4L2 indices: `v4l2-ctl --list-devices`
  - RTSP: use `source = "rtsp://user:pass@host:554/stream"` in the config.
- 4 GPIO LEDs (active-high, BCM numbering; configurable in `[leds]`):

  | Signal | BCM pin (default) | Color |
  |---|---|---|
  | IN green | 17 | ALLOW at entry |
  | IN red | 27 | REJECT at entry |
  | OUT green | 22 | ALLOW at exit |
  | OUT red | 23 | REJECT at exit |

  Current-limiting resistors (220-330 Ohm) in series; LED anodes to the
  pins, cathodes to GND.
- Optional HDMI/SPI framebuffer for the status display variant
  (`/dev/fb0`; `SDL_VIDEODRIVER=fbcon SDL_FBDEV=/dev/fb0`).

## 2. Install

```bash
# on the Pi, as root:
./scripts/install.sh headless    # or: display
```

The installer (idempotent) creates the `gate` system user, installs deps
(`libgpiod2`, venv), copies the repo to `/opt/gate`, installs the Python
venv, generates the web secret, writes an initial random admin password
(printed once to stdout), writes `/etc/gate/config.toml` (prompts for
camera sources), installs udev rules (gpio/video groups) and enables the
hardened systemd unit (`gate.service` or `gate-display.service`).

## 3. Bring-up checklist (manual, no cameras on dev workstation)

1. `systemctl status gate` -> active (running); `journalctl -u gate -f`
   shows camera retry logs when a camera is absent (expected: no crash).
2. Cameras: `v4l2-ctl --list-devices`; set `source` in
   `/etc/gate/config.toml` and restart: `systemctl restart gate`.
   Dashboard shows IN/OUT camera OK when frames flow.
3. Admin password: `sudo -u gate /opt/gate/venv/bin/python -m gate.cli passwd --config /etc/gate/config.toml`
   (or `gate_cli.py passwd`); the hash lands in `/var/lib/gate/admin.hash`.
4. Web UI: `http://<pi-ip>:8080/`, log in, dashboard shows System ONLINE,
   camera status, Database OK, vehicles inside, last event.
5. Register a plate: web UI Vehicles tab (plate canonical form, e.g.
   `29A1-678.90`) or
   `gate_cli.py add-vehicle 29A1-678.90 --note "resident"`.
6. Offline test without cameras/GPIO:
   `gate_cli.py simulate --video tests/fixtures/e2e_in.mp4 --side IN --config /etc/gate/config.toml`
   -> ALLOW event for the registered plate and `inside=1`.
7. LEDs: physically present plates at the IN camera -> green LED; unknown
   plate -> red LED. Rejected readings (unreadable/low confidence) blink
   red. Repeat at the OUT camera.
8. Event history + crops: `/var/lib/gate/images/YYYY/MM/DD/` (7-day
   retention, pruned daily by the watchdog).
9. Reboot the Pi; confirm the service comes back and the gate still works
   (auto-restart + camera auto-reconnect are exercised).

## 4. Behavior notes

- Failure of a camera: that side shows FAIL on the dashboard; the feed
  thread retries forever; the rest of the system keeps running.
- Failure of the GPIO chip/LEDs: logged and ignored - an LED fault never
  blocks the gate logic.
- Low-confidence or unreadable plates: red blink, REJECT/LOW_CONF event.
- Foreigner/diplomatic plates and car plates: rejected
  (INVALID_FORMAT / UNREGISTERED) - motorcycle-only gate.
- The watchdog restarts dead camera threads and checks the database every
  5 s; it never kills the process (systemd owns restart).

## 5. Updates (signed tags only)

`scripts/update.sh <tag>` fetches tags, **verifies the tag signature**
(requires `scripts/update_key.asc` - absent by default; add the
maintainer's key to enable updates), checks out the tag, reinstalls
requirements and restarts the service. No auto-update without signature
verification.

## 6. License / model notes

Runtime models in `/opt/gate/models`: `plate_det.onnx` (trained by this
project from scratch) and `ocr_rec.onnx` (PP-OCRv3, Apache-2.0). See
`LICENSES.md` / `models/README.md` in the repository.
