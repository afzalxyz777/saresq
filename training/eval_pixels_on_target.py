"""The pixels-on-target curve (Section 7.7): recall vs. person height in pixels.

    python training/eval_pixels_on_target.py \
        --weights-p3 results/detector/v8n_p3_hituav_640/weights/best.pt \
        --weights-p2 results/detector/v8n_p2_hituav_640/weights/best.pt \
        --data data/converted/hituav/hituav.yaml --split val

Writes ``results/detector/pixels_on_target.csv``. This is the measurement that
``eval_sweep_width.py`` integrates into the Koopman sweep width W, so it is not
an academic curve -- it is where the search model's central constant comes from.

Method: the validation images are progressively downsampled and padded back to
the original canvas, so the network sees *smaller people* rather than smaller
images. Recall is then binned by the ground-truth height each person ends up
at. Only the person class is scored.

Padding is the frame's own median, not black. On a thermal image black is not
neutral filler -- it reads as a large, very cold region, which is exactly the
kind of contrast the model was trained to find. Padding with the scene's median
keeps the background statistics of the padded area consistent with the real
image, so the measurement reflects object size and nothing else.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import cv2
import numpy as np
import yaml

# Run as a script from anywhere, and still be able to import sibling modules.
# Without this, `from training.train_detector import pick_device` inside main()
# raises ModuleNotFoundError -- at the END of argument parsing, i.e. only once
# the pipeline has already spent hours training the two models this step reads.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

SCALE_FACTORS = [1.0, 0.75, 0.5, 0.35, 0.25]
HEIGHT_BINS = [(2, 4), (4, 8), (8, 16), (16, 32), (32, 10_000)]
IOU_MATCH = 0.5
PERSON_CLASS = 0


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,))
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def read_yolo_labels(path: pathlib.Path, w: int, h: int) -> np.ndarray:
    """(N, 4) xyxy pixel boxes for the person class."""
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if int(float(parts[0])) != PERSON_CLASS:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


def shrink_scene(img: np.ndarray, scale: float) -> np.ndarray:
    """Downsample then pad back to the original canvas with the scene median."""
    if scale >= 1.0:
        return img
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    fill = np.median(img.reshape(-1, img.shape[2]), axis=0).astype(img.dtype)
    canvas = np.empty_like(img)
    canvas[:, :] = fill
    canvas[:nh, :nw] = small
    return canvas


def evaluate(model, image_paths, label_dir, scale, conf, batch, device):
    """(gt_height_px, matched) per person, at one scale."""
    records: list[tuple[float, bool]] = []
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start:start + batch]
        frames, truths = [], []
        for path in chunk:
            img = cv2.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt = read_yolo_labels(label_dir / (path.stem + ".txt"), w, h) * scale
            frames.append(shrink_scene(img, scale))
            truths.append(gt)
        if not frames:
            continue
        results = model.predict(frames, conf=conf, verbose=False, device=device)
        for result, gt in zip(results, truths):
            if result.boxes is not None and len(result.boxes):
                cls = result.boxes.cls.cpu().numpy()
                pred = result.boxes.xyxy.cpu().numpy()[cls == PERSON_CLASS]
            else:
                pred = np.zeros((0, 4))
            for row in gt:
                ious = iou_one_to_many(row, pred)
                records.append((float(row[3] - row[1]), bool((ious >= IOU_MATCH).any())))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--weights-p3", required=True)
    ap.add_argument("--weights-p2", required=True)
    ap.add_argument("--data", default=str(repo / "data" / "converted" / "hituav" / "hituav.yaml"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(repo / "results" / "detector" / "pixels_on_target.csv"))
    args = ap.parse_args()

    from ultralytics import YOLO
    from training.train_detector import pick_device

    device = pick_device(args.device)
    cfg = yaml.safe_load(open(args.data))
    img_dir = pathlib.Path(cfg["path"]) / cfg[args.split]
    label_dir = pathlib.Path(str(img_dir).replace("images", "labels"))
    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if args.limit:
        image_paths = image_paths[: args.limit]

    rows = []
    for variant, weights in (("p3", args.weights_p3), ("p2", args.weights_p2)):
        model = YOLO(weights)
        for scale in SCALE_FACTORS:
            records = evaluate(model, image_paths, label_dir, scale, args.conf, args.batch, device)
            for lo, hi in HEIGHT_BINS:
                in_bin = [m for h, m in records if lo <= h < hi]
                label = f"{lo}-{hi if hi < 10_000 else 'inf'}"
                if not in_bin:
                    print(f"{variant} scale={scale} bin={label} n=0 (skipped)")
                    continue
                recall = sum(in_bin) / len(in_bin)
                rows.append({"variant": variant, "scale": scale, "bin_px": label,
                             "n": len(in_bin), "recall": round(recall, 4)})
                print(f"{variant} scale={scale} bin={label:>8s} n={len(in_bin):5d} recall={recall:.3f}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["variant", "scale", "bin_px", "n", "recall"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
