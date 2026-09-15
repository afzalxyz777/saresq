"""Can a stock COCO detector serve as the RGB confirmation branch?

    python training/eval_rgb_confirm.py --visdrone <VisDrone2019-DET-val dir>

Writes ``results/detector/rgb_confirm.csv`` (one row per person instance) and
``results/detector/rgb_confirm.json`` (the binned curve).

Background. The RGB *fine-tune* on RGBTDronePerson could never work -- those
visible frames are night images with no person signal (see
docs/deep_learning.md). That left ``p_rgb``/``has_rgb``/``a_rgb`` identically
zero across all 11,144 fusion rows, and the fusion head duly gave them zero
weight. The conclusion drawn at the time was "no visible branch". This script
tests the cheaper hypothesis that was never tried: **a COCO-pretrained
detector, with no fine-tuning at all, may already be good enough** -- because
the crop path is not a small-object problem.

Method. The shipped pipeline never runs a detector on a full frame: the
thermal gate fires, and ``detector.crop_px`` pixels are cropped around the
hit. So full-frame AP is the wrong measurement -- it scores the model on
finding people, which is the *gate's* job, not this branch's. What matters is:
given the gate has already pointed at a person, does the RGB detector confirm
it? That is recall on gate-style crops, and it is measured here.

Scale is the whole story, so recall is binned by sqrt(w*h) rather than by box
height. VisDrone's oblique street views make a pedestrian tall and narrow;
this payload looks straight down, which makes the same person short and wide.
Height is therefore not comparable across the two geometries, and sqrt(area)
is. At 20-30 m this payload's own optics produce (docs/deep_learning.md):

    prone adult    115 x 31 px  -> sqrt(area) 60 px
    standing adult  31 x 24 px  -> sqrt(area) 27 px

Read those two rows of the output table. The overall figure is dominated by
VisDrone's median 25 px pedestrians and describes VisDrone, not this payload.

Night frames are excluded by mean brightness: this measures the daytime
branch, which is the only condition in which it would ever be consulted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parent.parent
# VisDrone DET: left,top,w,h,score,category,truncation,occlusion
# category 1 = pedestrian, 2 = people; score 0 marks an ignored region.
PERSON_CATS = (1, 2)
BINS = [(0, 16), (16, 24), (24, 32), (32, 48), (48, 72), (72, 10**6)]


def _gt_boxes(ann: pathlib.Path) -> list[tuple[int, int, int, int]]:
    out = []
    for line in ann.read_text().split("\n"):
        parts = line.strip().rstrip(",").split(",")
        if len(parts) < 6:
            continue
        x, y, w, h, score, cat = (int(float(v)) for v in parts[:6])
        if score == 0 or cat not in PERSON_CATS or w <= 0 or h <= 0:
            continue
        out.append((x, y, w, h))
    return out


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visdrone", required=True,
                    help="VisDrone2019-DET-val directory (images/, annotations/)")
    ap.add_argument("--weights", default="yolov8n.pt", help="stock COCO weights")
    ap.add_argument("--crop", type=int, default=160, help="detector.crop_px")
    ap.add_argument("--conf", type=float, default=0.15, help="detector.conf")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--dark", type=float, default=60.0,
                    help="mean brightness below this = night frame, excluded")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-csv", default=str(REPO / "results/detector/rgb_confirm.csv"))
    ap.add_argument("--out-json", default=str(REPO / "results/detector/rgb_confirm.json"))
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    root = pathlib.Path(args.visdrone)
    if not (root / "annotations").is_dir():
        raise SystemExit(f"not a VisDrone DET split: {root}")
    model = YOLO(args.weights)

    rows: list[dict] = []
    n_day = n_night = 0
    for ann in sorted((root / "annotations").glob("*.txt")):
        img_p = root / "images" / (ann.stem + ".jpg")
        if not img_p.exists():
            continue
        im = Image.open(img_p).convert("RGB")
        if float(np.asarray(im.convert("L"), dtype=np.float32).mean()) < args.dark:
            n_night += 1
            continue
        n_day += 1
        W, H = im.size
        for (x, y, w, h) in _gt_boxes(ann):
            cx, cy = x + w / 2, y + h / 2
            x0 = int(min(max(cx - args.crop / 2, 0), max(W - args.crop, 0)))
            y0 = int(min(max(cy - args.crop / 2, 0), max(H - args.crop, 0)))
            crop = im.crop((x0, y0, x0 + args.crop, y0 + args.crop))
            pred = model.predict(crop, imgsz=args.crop, conf=args.conf,
                                 classes=[0], verbose=False, device=args.device)[0]
            target = (x - x0, y - y0, x + w - x0, y + h - y0)
            best = 0.0
            best_conf = 0.0
            for box, cf in zip(pred.boxes.xyxy.cpu().numpy(),
                               pred.boxes.conf.cpu().numpy()):
                v = _iou(target, tuple(float(t) for t in box))
                if v > best:
                    best, best_conf = v, float(cf)
            rows.append({"image": img_p.name, "w": w, "h": h,
                         "sqrt_area": round(math.sqrt(w * h), 2),
                         "best_iou": round(best, 4),
                         "conf": round(best_conf, 4),
                         "hit": int(best >= args.iou)})

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    curve = []
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["sqrt_area"] < hi]
        if sel:
            curve.append({"lo": lo, "hi": None if hi > 10**5 else hi,
                          "n": len(sel),
                          "recall": sum(r["hit"] for r in sel) / len(sel)})
    summary = {
        "weights": args.weights, "crop_px": args.crop, "conf": args.conf,
        "daytime_frames": n_day, "night_frames_excluded": n_night,
        "instances": len(rows),
        "median_sqrt_area": statistics.median(r["sqrt_area"] for r in rows),
        "overall_recall": sum(r["hit"] for r in rows) / len(rows),
        "curve": curve,
    }
    pathlib.Path(args.out_json).write_text(json.dumps(summary, indent=2))

    print(f"daytime frames {n_day} (excluded {n_night} night), "
          f"{len(rows)} person instances, "
          f"median sqrt(area) {summary['median_sqrt_area']:.0f} px")
    print("\n  sqrt(area)      n     recall @ IoU0.5")
    for c in curve:
        lbl = f"{c['lo']}-{c['hi']}" if c["hi"] else f"{c['lo']}+"
        print(f"  {lbl:>10}  {c['n']:6d}      {c['recall']:.3f}")
    print(f"\n  {'overall':>10}  {len(rows):6d}      {summary['overall_recall']:.3f}")
    print(f"\nwrote {args.out_csv} and {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
