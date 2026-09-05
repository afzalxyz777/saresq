"""The pixels-on-target curve (Section 7.7): recall vs. person height in pixels,
for both head variants, at five downsampling factors.

    python training/eval_pixels_on_target.py \
        --weights-p3 runs/detect/v8n_p3_hituav_640/weights/best.pt \
        --weights-p2 runs/detect/v8n_p2_hituav_640/weights/best.pt \
        --data data/hituav.yaml --split val

Writes results/pixels_on_target.csv and results/pixels_on_target.png.
"""
from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

SCALE_FACTORS = [1.0, 0.75, 0.5, 0.35, 0.25]
HEIGHT_BINS = [(2, 4), (4, 8), (8, 16), (16, 32), (32, 10_000)]
CONF_THRESH = 0.25
IOU_MATCH = 0.5


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    iw, ih = np.clip(ix2 - ix1, 0, None), np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-9, None)


def _read_yolo_labels(label_path: pathlib.Path, img_w: int, img_h: int) -> np.ndarray:
    """Returns Nx5 array [class, x1, y1, x2, y2] in pixel coords."""
    if not label_path.exists():
        return np.zeros((0, 5))
    rows = []
    for line in label_path.read_text().strip().splitlines():
        cls, cx, cy, w, h = map(float, line.split())
        x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        x2, y2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
        rows.append([cls, x1, y1, x2, y2])
    return np.array(rows) if rows else np.zeros((0, 5))


def evaluate_model_at_scale(model: YOLO, image_paths: list[pathlib.Path], label_dir: pathlib.Path, scale: float):
    """Returns list of (gt_height_px_after_scaling, matched: bool) for the person class (0)."""
    records = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        gt = _read_yolo_labels(label_dir / (img_path.stem + ".txt"), w0, h0)
        gt_person = gt[gt[:, 0] == 0] if len(gt) else gt

        new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Pad back to original size so the network sees smaller people, not smaller images.
        canvas = np.zeros_like(img)
        canvas[:new_h, :new_w] = small
        gt_scaled = gt_person.copy()
        if len(gt_scaled):
            gt_scaled[:, 1:] *= scale

        result = model.predict(canvas, conf=CONF_THRESH, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
        pred_cls = result.boxes.cls.cpu().numpy() if result.boxes is not None else np.zeros((0,))
        pred_person = pred_boxes[pred_cls == 0] if len(pred_boxes) else pred_boxes

        for row in gt_scaled:
            gt_h = row[4] - row[2]
            matched = False
            if len(pred_person):
                ious = _iou_xyxy(row[1:5][None, :], pred_person)[0]
                matched = bool((ious >= IOU_MATCH).any())
            records.append((gt_h, matched))
    return records


def recall_per_bin(records: list[tuple[float, bool]]) -> dict[str, float]:
    out = {}
    for lo, hi in HEIGHT_BINS:
        in_bin = [m for h, m in records if lo <= h < hi]
        out[f"{lo}-{hi if hi < 10_000 else 'inf'}"] = (sum(in_bin) / len(in_bin)) if in_bin else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-p3", required=True)
    ap.add_argument("--weights-p2", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data))
    img_dir = pathlib.Path(cfg["path"]) / cfg[args.split]
    label_dir = pathlib.Path(str(img_dir).replace("images", "labels"))
    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    models = {"p3": YOLO(args.weights_p3), "p2": YOLO(args.weights_p2)}
    rows = []
    for variant, model in models.items():
        for scale in SCALE_FACTORS:
            records = evaluate_model_at_scale(model, image_paths, label_dir, scale)
            bins = recall_per_bin(records)
            for bin_name, recall in bins.items():
                rows.append({"variant": variant, "scale": scale, "bin_px": bin_name, "recall": recall})
                print(f"{variant} scale={scale} bin={bin_name} recall={recall:.3f}" if recall == recall else
                      f"{variant} scale={scale} bin={bin_name} recall=n/a (empty bin)")

    pathlib.Path("results").mkdir(exist_ok=True)
    import csv
    with open("results/pixels_on_target.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "scale", "bin_px", "recall"])
        writer.writeheader()
        writer.writerows(rows)
    print("wrote results/pixels_on_target.csv")


if __name__ == "__main__":
    main()
