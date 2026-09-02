"""Vision package: ONNX plate detection, line splitting, OCR, pipeline.

Model loads are lazy singletons (one session per process). Sessions use
``CPUExecutionProvider`` with ``intra_op_num_threads=1`` (Raspberry Pi
budget; onnxruntime is thread-limited anyway at this size).
"""
