# Licenses

- **PP-OCRv3 recognition model** (`models/ocr_rec.onnx`): Apache-2.0, from
  PaddleOCR upstream. Full text in `models/LICENSE-PPOCR.txt`; upstream URL in
  `models/README.md`.
- **Plate detector ONNX** (`models/plate_det.onnx`): trained by this project
  **from scratch** (YOLOv8n architecture, no pretrained-weight initialization)
  on the winter2897 Vietnamese plate dataset. No weight-derivative license
  exposure from the architecture or the training data.
- **winter2897 dataset** (training only, never redistributed): no license file
  upstream. See `models/LICENSE-DATASET.txt`. Dataset images are not committed
  to this repository.
- **Ultralytics (AGPL-3.0)**: applies to the training toolchain only, never to
  the runtime — the gate runtime is plain onnxruntime + OpenCV, no Ultralytics
  dependency.
