# VN Plate Gate

Offline, edge-only motorcycle access-control gate for Vietnamese plate numbers.

- IN/OUT cameras (V4L2 or RTSP) with motion-triggered capture
- Vietnamese plate detection (trained YOLOv8n ONNX) and OCR (PP-OCRv3 ONNX)
- Format normalization/validation, local SQLite registry with inside/outside state
- Event classification: ALLOW / REJECT (unregistered, invalid format, already inside/outside, low confidence)
- GPIO LEDs per side (green/red), auto-recovering camera threads, watchdog
- Hardened non-root systemd service, password-protected LAN admin web UI
- Two variants: headless and framebuffer status display (SDL)

Development and tests run on x86_64; deployment targets Raspberry Pi Zero 2 W
(Raspberry Pi OS Bookworm 64-bit). See `docs/DEPLOY.md` for hardware wiring,
install, and bring-up.
