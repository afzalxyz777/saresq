"""Export a trained detector to INT8 TFLite at each crop size (Section 7.8).

    python training/export_tflite.py --weights runs/detect/v8n_p2_hituav_640/weights/best.pt \
        --data data/hituav.yaml --sizes 160 224 640

Run on the laptop/Colab, never on the Pi (routes through ONNX + onnx2tf).
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="dataset YAML, source of calibration images")
    ap.add_argument("--sizes", type=int, nargs="+", default=[160, 224, 640])
    args = ap.parse_args()

    model = YOLO(args.weights)
    for size in args.sizes:
        path = model.export(format="tflite", int8=True, imgsz=size, data=args.data)
        print(f"exported imgsz={size} -> {path}")


if __name__ == "__main__":
    main()
