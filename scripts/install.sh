#!/usr/bin/env bash
# Idempotent installer for the gate (run as root):
#   ./install.sh headless   -> systemd/gate.service
#   ./install.sh display    -> systemd/gate-display.service (framebuffer)
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT="${1:-headless}"
case "$VARIANT" in
  headless) UNIT=gate ;;
  display)  UNIT=gate-display ;;
  *) echo "usage: $0 [headless|display]" >&2; exit 2 ;;
esac

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo ./scripts/install.sh $VARIANT" >&2
  exit 1
fi

echo "==> installing apt dependencies"
apt-get update -qq
apt-get install -y -qq libgpiod2 python3-venv ffmpeg 2>/dev/null || \
  apt-get install -y -qq libgpiod2 python3-venv

echo "==> creating gate user (system, no shell)"
id gate >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin gate

echo "==> directories"
mkdir -p /opt/gate /var/lib/gate/images /etc/gate

echo "==> copying repository to /opt/gate"
rsync -a --delete \
  --exclude '.venv*' --exclude 'models/vendor' --exclude 'data' \
  --exclude 'runs' --exclude '.git' --exclude '__pycache__' \
  ./ /opt/gate/ 2>/dev/null || {
    # rsync absent: plain copy minus excludes
    rm -rf /opt/gate.new && mkdir -p /opt/gate.new
    cp -r gate gate_app.py gate_cli.py requirements.txt pyproject.toml \
      models systemd deploy /opt/gate.new/
    rm -rf /opt/gate && mv /opt/gate.new /opt/gate
  }

echo "==> python venv + dependencies"
[ -d /opt/gate/venv ] || python3 -m venv /opt/gate/venv
/opt/gate/venv/bin/pip install --quiet --upgrade pip
/opt/gate/venv/bin/pip install --quiet -r /opt/gate/requirements.txt

echo "==> web secret"
if [ ! -f /var/lib/gate/secret ]; then
  head -c 32 /dev/urandom > /var/lib/gate/secret
  chmod 600 /var/lib/gate/secret
fi

echo "==> config template"
if [ ! -f /etc/gate/config.toml ]; then
  echo "camera IN source (V4L2 index or RTSP URL) [0]:"
  read -r CAM_IN
  echo "camera OUT source (V4L2 index or RTSP URL) [1]:"
  read -r CAM_OUT
  CAM_IN="${CAM_IN:-0}"
  CAM_OUT="${CAM_OUT:-1}"
  cat > /etc/gate/config.toml <<EOF
[cameras.in]
source = "${CAM_IN}"
width = 640
height = 360
fps = 10

[cameras.out]
source = "${CAM_OUT}"
width = 640
height = 360
fps = 10

[vision]
burst_frames = 5
min_frames_detected = 2
min_confidence = 0.75
model_dir = "/opt/gate/models"

[decision]
cooldown_s = 10.0

[leds]
in_green = 17
in_red = 27
out_green = 22
out_red = 23
allow_s = 2.0
reject_s = 2.0
blink_s = 0.2

[storage]
db_path = "/var/lib/gate/gate.db"
images_dir = "/var/lib/gate/images"
retention_days = 7

[web]
host = "0.0.0.0"
port = 8080
secret_file = "/var/lib/gate/secret"
password_file = "/var/lib/gate/admin.hash"

[display]
enabled = false
width = 640
height = 480
EOF
  chmod 640 /etc/gate/config.toml
  chown root:gate /etc/gate/config.toml
fi

echo "==> initial admin password (printed once)"
if [ ! -f /var/lib/gate/admin.hash ]; then
  PASSWORD="$(head -c 9 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 12)"
  cd /opt/gate
  printf '%s\n%s\n' "$PASSWORD" "$PASSWORD" | \
    ./venv/bin/python -m gate.cli passwd --config /etc/gate/config.toml >/dev/null
  echo "ADMIN PASSWORD: $PASSWORD  (change with: gate_cli.py passwd)"
fi

echo "==> ownership"
chown -R gate:gate /var/lib/gate /opt/gate

echo "==> udev rules"
install -m 644 deploy/99-gate-udev.rules /etc/udev/rules.d/99-gate-udev.rules
udevadm control --reload-rules || true
udevadm trigger || true

echo "==> systemd unit"
install -m 644 "systemd/${UNIT}.service" "/etc/systemd/system/${UNIT}.service"
systemctl daemon-reload
systemctl enable --now "${UNIT}.service"

echo "install complete (${UNIT})."
echo "Manual bring-up checklist: see docs/DEPLOY.md"
