"""What INT8 quantisation actually costs (the eval that was missing).

    ./.venv-tf/bin/python training/eval_int8_gap.py \
        --weights results/detector/v8n_p3_hituav_640/weights/best.pt \
        --tflite  models/yolov8n_p3_640_w8a32.tflite \
        --data    data/converted/hituav/hituav.yaml --split val

Writes ``results/detector/int8_gap.json``.

Everything downstream of the detector -- the fusion ledger, the sweep width, the
whole deck -- quotes Person AP50 0.821. That number was measured on the FLOAT32
PyTorch model. The thing that flies is a 3.3 MB INT8 TFLite file. Nobody had
ever measured whether those are the same model, and quantisation error does not
announce itself: it shifts scores by a few percent and silently drops the
faintest detections, which on this dataset are exactly the people at 8-16 px
that matter most. Shipping an unmeasured 8-bit approximation of your headline
number is how a demo dies on stage.

## Two measurements, because they answer different questions

**1. Numerical fidelity.** Identical preprocessed tensors go into both models and
the raw head outputs are compared directly -- correlation and mean absolute error
on the box rows and on the class rows, separately. This isolates quantisation
with nothing else in the path: no NMS, no thresholding, no letterbox difference.
If this is clean the export is faithful, whatever the task numbers say.

**2. Task accuracy.** Person AP50 on the validation split, computed for both
models through the SAME letterbox and the SAME decode/NMS. This is the number
that matters, and sharing the decode is what makes it a measurement of
quantisation rather than a measurement of two different postprocessing stacks.
Running Ultralytics' own validator against our runtime would have confounded the
two -- a gap could have come from their NMS defaults rather than from INT8.

Box rows and class rows are reported separately on purpose. They quantise very
differently: box regressions are smooth and bounded, class logits are peaky, and
a model can hold its boxes while losing its faint positives. One blended error
figure would hide precisely the failure this script exists to catch.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import yaml

# Run as a script from anywhere and still import the saresq package. Without
# this the import below fails only after the models have loaded.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from saresq.detect.decode import decode_and_nms          # noqa: E402
from saresq.detect.tflite_detector import letterbox      # noqa: E402
from saresq.detect.runtime import TFLiteModel            # noqa: E402

PERSON_CLASS = 0
IOU_MATCH = 0.5


def read_person_labels(label_path: pathlib.Path, w: int, h: int) -> np.ndarray:
    """(N, 4) xyxy pixel boxes for the person class, from a YOLO .txt label."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) != PERSON_CLASS:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(out, dtype=np.float32).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def average_precision(records: list, n_gt: int) -> float:
    """All-point interpolated AP from (score, is_true_positive) records.

    All-point rather than the 11-point or 101-point variants because those
    sample a curve that is already exact here; interpolating a curve we hold in
    full would only add quantisation of its own to a script about quantisation.
    """
    if n_gt == 0 or not records:
        return 0.0
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records], dtype=np.float64)
    fp = np.cumsum([not r[1] for r in records], dtype=np.float64)
    recall = tp / n_gt
    precision = tp / np.clip(tp + fp, 1e-9, None)
    # Monotonic envelope, then integrate over recall.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    r = np.concatenate(([0.0], recall))
    p = np.concatenate(([precision[0] if len(precision) else 0.0], precision))
    return float(np.sum(np.diff(r) * p[1:]))


def score_one(preds: np.ndarray, scores: np.ndarray, gt: np.ndarray,
              records: list) -> int:
    """Greedy highest-score-first matching at IoU 0.5. Returns GT count."""
    order = np.argsort(-scores)
    ious = iou_matrix(preds[order], gt) if len(preds) else np.zeros((0, len(gt)))
    taken = set()
    for i in range(len(order)):
        best, best_j = 0.0, -1
        for j in range(len(gt)):
            if j in taken:
                continue
            if ious[i, j] > best:
                best, best_j = ious[i, j], j
        hit = best >= IOU_MATCH
        if hit:
            taken.add(best_j)
        records.append((float(scores[order[i]]), hit))
    return len(gt)


def list_split_images(data_yaml: pathlib.Path, split: str) -> list[pathlib.Path]:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = pathlib.Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    entry = cfg[split]
    sub = root / (entry[0] if isinstance(entry, list) else entry)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in sub.rglob("*") if p.suffix.lower() in exts)


def label_for(image_path: pathlib.Path) -> pathlib.Path:
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return pathlib.Path(*parts).with_suffix(".txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250,
                    help="images to score. 250 is enough to separate a healthy "
                         "export from a broken one; the full split adds runtime, "
                         "not confidence.")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="deliberately low: AP integrates the whole "
                         "precision-recall curve, so a high threshold would "
                         "truncate it and flatter both models equally.")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="results/detector/int8_gap.json")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    tfl = TFLiteModel(args.tflite, num_threads=4)
    size = tfl.imgsz
    yolo = YOLO(args.weights)
    torch_model = yolo.model.float().eval()

    images = list_split_images(pathlib.Path(args.data), args.split)[:args.limit]
    if not images:
        raise SystemExit(f"no images found for split {args.split!r}")
    print(f"  model     {pathlib.Path(args.tflite).name}  input {size}x{size}")
    print(f"  scoring   {len(images)} images from {args.split}\n")

    # Probe the box convention once, exactly as the deployment runtime does.
    grey = np.full((1, size, size, 3), 114, dtype=np.uint8)
    peak = float(np.max(np.abs(np.squeeze(tfl.infer_one(grey))[:4])))
    tfl_normalised = peak <= max(2.0, size * 0.05)

    box_corr, cls_corr, box_mae, cls_mae = [], [], [], []
    rec_f, rec_q, n_gt = [], [], 0

    for k, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_person_labels(label_for(img_path), w, h)

        padded, gain, pad_x, pad_y = letterbox(img, size)

        # --- float32 reference -------------------------------------------
        x = torch.from_numpy(padded[..., ::-1].copy()).permute(2, 0, 1)[None]
        x = x.float() / 255.0
        with torch.no_grad():
            out = torch_model(x)
        raw_f = (out[0] if isinstance(out, (list, tuple)) else out).cpu().numpy()

        # --- INT8 deployed path ------------------------------------------
        raw_q = tfl.infer_one(padded[None, ...])

        yf = np.squeeze(raw_f)
        yq = np.squeeze(raw_q)
        if yf.shape[0] > yf.shape[1]:
            yf = yf.T
        if yq.shape[0] > yq.shape[1]:
            yq = yq.T
        if tfl_normalised:
            yq = yq.copy()
            yq[:4] *= size

        if yf.shape == yq.shape:
            for rows, corr, mae in ((slice(0, 4), box_corr, box_mae),
                                    (slice(4, None), cls_corr, cls_mae)):
                a, b = yf[rows].ravel(), yq[rows].ravel()
                mae.append(float(np.mean(np.abs(a - b))))
                if a.std() > 1e-9 and b.std() > 1e-9:
                    corr.append(float(np.corrcoef(a, b)[0, 1]))

        # --- same decode for both, so only quantisation differs ------------
        for y, records in ((yf, rec_f), (yq, rec_q)):
            boxes, scores, classes = decode_and_nms(y, conf=args.conf, iou=args.iou)
            keep = classes == PERSON_CLASS
            boxes, scores = boxes[keep], scores[keep]
            if len(boxes):
                boxes = np.column_stack([(boxes[:, 0] - pad_x) / gain,
                                         (boxes[:, 1] - pad_y) / gain,
                                         (boxes[:, 2] - pad_x) / gain,
                                         (boxes[:, 3] - pad_y) / gain])
            score_one(boxes, scores, gt, records)
        n_gt += len(gt)

        if (k + 1) % 50 == 0:
            print(f"    {k + 1}/{len(images)} images, {n_gt} people so far")

    ap_f = average_precision(rec_f, n_gt)
    ap_q = average_precision(rec_q, n_gt)
    gap = ap_q - ap_f
    rel = (gap / ap_f * 100.0) if ap_f > 0 else float("nan")

    result = {
        "tflite": args.tflite, "weights": args.weights, "imgsz": size,
        "images": len(images), "gt_person": n_gt,
        "person_AP50_float32": round(ap_f, 5),
        "person_AP50_int8": round(ap_q, 5),
        "absolute_gap": round(gap, 5),
        "relative_gap_pct": round(rel, 2),
        "box_rows_correlation": round(float(np.mean(box_corr)), 5) if box_corr else None,
        "cls_rows_correlation": round(float(np.mean(cls_corr)), 5) if cls_corr else None,
        "box_rows_mae_px": round(float(np.mean(box_mae)), 4) if box_mae else None,
        "cls_rows_mae": round(float(np.mean(cls_mae)), 5) if cls_mae else None,
    }

    print("\n  NUMERICAL FIDELITY (identical inputs, raw head outputs)")
    print(f"    box rows   corr {result['box_rows_correlation']}   MAE {result['box_rows_mae_px']} px")
    print(f"    cls rows   corr {result['cls_rows_correlation']}   MAE {result['cls_rows_mae']}")
    print("\n  TASK ACCURACY (same letterbox, same decode, same NMS)")
    print(f"    Person AP50  float32 {ap_f:.4f}")
    print(f"    Person AP50  INT8    {ap_q:.4f}")
    print(f"    gap                  {gap:+.4f}  ({rel:+.1f}%)")

    # Verdict. A few percent of AP is normal for INT8; a large drop means the
    # calibration set did not cover the activation range, not that the model is
    # bad. Two guards matter here:
    #
    # 1. If the FLOAT model also scores ~0 the comparison is uninformative, not
    #    a failure -- it means the objects are too small to detect at this input
    #    size at all. Whole-frame AP at 160 or 224 does exactly that on HIT-UAV:
    #    a 640x512 frame letterboxed to 160 leaves people 2-4 px. Those models
    #    exist to run on GATE CROPS, so score them at 640 or not at all. Without
    #    this guard `abs(nan) <= 5` is False and the script cries wolf.
    # 2. Class-row correlation is the more robust signal anyway, because it is
    #    measured on raw outputs with nothing else in the path. Faint positives
    #    are the first thing INT8 loses, and they live in the class rows.
    cls_corr = result["cls_rows_correlation"]
    if ap_f < 0.01:
        verdict = (f"AP is ~0 for BOTH models at {size}px -- objects are too small "
                   f"to detect at this input size, so the AP comparison says "
                   f"nothing. Judge this export by numerical fidelity "
                   f"(cls corr {cls_corr}), and score whole frames at 640.")
    elif abs(rel) <= 5:
        verdict = f"INT8 export is faithful ({rel:+.1f}% AP, cls corr {cls_corr})"
    else:
        verdict = (f"INT8 export lost real accuracy ({rel:+.1f}% AP) -- recheck "
                   f"the calibration set covers the deployed activation range")
    if cls_corr is not None and cls_corr < 0.90:
        verdict += f" | WARNING: class-row correlation {cls_corr} is low"
    print(f"\n  VERDICT: {verdict}")
    result["verdict"] = verdict

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
