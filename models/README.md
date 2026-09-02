# Model files

## models/plate_det.onnx - plate detector

- Architecture: YOLOv8n, trained **from scratch** (`model=yolov8n.yaml`,
  `pretrained=False`; no pretrained-weight initialization, so no
  Ultralytics/AGPL weight-derivative applies - AGPL covers the training
  toolchain only, never the runtime).
- Training data: winter2897 Vietnamese plate dataset (images of cars and
  motorcycles), 90/10 train/val split (fixed seed 42), 7434 train / 825
  val images. Dataset used for training only; NOT redistributed (see
  LICENSE-DATASET.txt).
- Training: 300-epoch schedule, patience 50, batch 32, imgsz 640, GPU
  (CUDA). Early-stopped at epoch 43: beyond it the val mAP50 moved only at
  noise level (~+0.00002/epoch) - the acceptance gate was met with a wide
  margin long before.
- **Recorded validation (best.pt): mAP50 = 0.995, mAP50-95 = 0.718,
  precision 0.992, recall 0.988** (ultralytics `yolo detect val`).
- ONNX export: opset 12, imgsz 640. Input `[1, 3, 640, 640]` NCHW float32
  (BGR image, letterboxed, /255); output `[1, 5, 8400]` = (cx, cy, w, h,
  score) per anchor in letterboxed pixel space. Decode in
  `gate/vision/detector.py` matches this exactly (no NMS in the graph).

## models/ocr_rec.onnx - PP-OCRv3 English recognizer

- Source: PaddleOCR `en_PP-OCRv3_rec_infer` (Apache-2.0; see
  LICENSE-PPOCR.txt). The Paddle-format tar was converted with paddle2onnx.
- Input `[N, 3, 48, W]` NCHW float32 (grayscale repeated 3x, resize to
  height 48 keeping aspect, width floor-8 capped at 320, mean/std 0.5
  normalization); output `[N, T, 97]` logits per timestep.
- Decode dict: 97 classes; recovered empirically by glyph probing (the
  model ships no dict and PaddleOCR's 37-char en_dict.txt does not match):
  class 0 = CTC blank, 1..10 = '0'..'9', 18..43 = 'A'..'Z', 50..75 =
  'a'..'z', remainder punctuation/accents decoded as dropped. Details and
  the lowercase-emission note live in `gate/vision/ocr.py`.

## Checksums

`SHA256SUMS` covers every file under `models/`.
