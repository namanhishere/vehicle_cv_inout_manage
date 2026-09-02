#!/usr/bin/env bash
# Fetch the PP-OCRv3 English recognition model (Apache-2.0, PaddleOCR).
# Idempotent: skips download when the final models/ocr_rec.onnx exists.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f models/ocr_rec.onnx ]; then
  echo "models/ocr_rec.onnx already present"
  exit 0
fi

mkdir -p models/vendor models
TAR=models/vendor/en_PP-OCRv3_rec_infer.tar
if [ ! -f "$TAR" ]; then
  curl -fL -o "$TAR" \
    https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar
fi
# the tar ships a directory (e.g. en_PP-OCRv3_rec_infer/) - locate its files
VENDOR_DIR=$(find models/vendor -name inference.pdmodel -o -name inference.onnx | head -1)
if [ -z "$VENDOR_DIR" ]; then
  echo "unexpected model tar layout in $TAR" >&2
  exit 1
fi
case "$VENDOR_DIR" in
  *.onnx)
    cp "$VENDOR_DIR" models/ocr_rec.onnx
    ;;
  *.pdmodel)
    # pinned fallback: convert Paddle format with paddle2onnx
    python3 -m venv .venv-conv
    .venv-conv/bin/pip install --quiet paddle2onnx
    .venv-conv/bin/paddle2onnx --model_dir "$(dirname "$VENDOR_DIR")" \
      --model_filename inference.pdmodel \
      --params_filename inference.pdiparams \
      --save_file models/ocr_rec.onnx
    ;;
esac
echo "OCR model ready: models/ocr_rec.onnx"
