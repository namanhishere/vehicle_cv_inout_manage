#!/usr/bin/env bash
# Download the winter2897 Vietnamese license plate dataset (training only,
# never redistributed) and prepare data/data.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data
if [ ! -f data/vn_plates.zip ]; then
  python3 -m venv .venv-train
  .venv-train/bin/pip install --quiet gdown
  .venv-train/bin/gdown \
    "https://drive.google.com/file/d/1KLK-DWgT3VoQH4fcTxAt2eB3sm7DGWAf/view?usp=sharing" \
    -O data/vn_plates.zip
fi
unzip -q -o data/vn_plates.zip -d data/
echo "dataset extracted under data/ - inspect layout, then write data/data.yaml"
