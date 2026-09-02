#!/usr/bin/env bash
# Train the plate detector from scratch (YOLOv8n architecture, no pretrained
# init -> weights are legally ours) and export to ONNX.
#
# Acceptance gate: validation mAP50 >= 0.90 (recorded in models/README.md).
# GPU: device=0; fall back to CPU by removing device=0 (much slower).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f data/data.yaml ]; then
  echo "data/data.yaml missing - run scripts/download_dataset.sh first" >&2
  exit 1
fi

.venv-train/bin/pip install --quiet torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
.venv-train/bin/pip install --quiet ultralytics

.venv-train/bin/yolo detect train model=yolov8n.yaml data=data/data.yaml \
  imgsz=640 epochs=300 patience=50 batch=32 device=0
.venv-train/bin/yolo detect val \
  model=runs/detect/train/weights/best.pt data=data/data.yaml
.venv-train/bin/yolo export model=runs/detect/train/weights/best.pt \
  format=onnx opset=12 imgsz=640
cp runs/detect/train/weights/best.onnx models/plate_det.onnx
echo "detector ONNX ready: models/plate_det.onnx"
