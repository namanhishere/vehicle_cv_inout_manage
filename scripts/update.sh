#!/usr/bin/env bash
# Signed-tag update: gate/update.sh <tag>
# Refuses to run without a trusted signing key (secure by default).
set -euo pipefail

TAG="${1:-}"
if [ -z "$TAG" ]; then
  echo "usage: sudo ./scripts/update.sh <git-tag>" >&2
  exit 2
fi

cd /opt/gate
if [ ! -f scripts/update_key.asc ]; then
  echo "error: no signing key at /opt/gate/scripts/update_key.asc" >&2
  echo "add the maintainer's public key to enable signed updates:" >&2
  echo "  gpg --import update_key.asc  (as the gate admin)" >&2
  exit 3
fi

git fetch --tags
git verify-tag "$TAG"

git checkout "$TAG"
[ -d venv ] || python3 -m venv venv
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt

UNIT=gate
systemctl restart "${UNIT}.service"
echo "updated to $TAG and restarted ${UNIT}."
