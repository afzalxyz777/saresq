"""Train the survivor detector: baseline (P3-P5) and P2-P5 variant (Section 7.4, 7.6).

Run on Colab (T4) or locally with device="mps" for a sanity check:

    python training/train_detector.py --variant p3 --data data/hituav.yaml --epochs 100
    python training/train_detector.py --variant p2 --data data/hituav.yaml --epochs 100

Requires the [training] extra: pip install -e ".[training]"
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def build_model(variant: str) -> YOLO:
    if variant == "p3":
        return YOLO("yolov8n.pt")
    if variant == "p2":
        return YOLO("yolov8n-p2.yaml").load("yolov8n.pt")
    raise ValueError(f"unknown variant {variant!r}, expected 'p3' or 'p2'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["p3", "p2"], required=True)
    ap.add_argument("--data", default="data/hituav.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 'mps' for a local Apple Silicon sanity run")
    ap.add_argument("--project", default="runs/detect")
    args = ap.parse_args()

    model = build_model(args.variant)
    model.info()  # records layers/params/GFLOPs for the p2-vs-p3 comparison table

    model.train(
        data=args.data,
        imgsz=args.imgsz,          # native HIT-UAV size; do not upscale
        epochs=args.epochs,
        batch=args.batch,
        patience=25,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=3,
        # Augmentation for nadir aerial imagery (Correction C9): any
        # rotation/flip is a valid view when the camera points straight down.
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,       # altitude varies -> aggressive scale jitter
        translate=0.1,
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,  # thermal has no hue; keep brightness jitter
        device=args.device,
        project=args.project,
        name=f"v8n_{args.variant}_hituav_{args.imgsz}",
        save_period=1,  # checkpoint every epoch: Colab sessions die
    )


if __name__ == "__main__":
    main()
