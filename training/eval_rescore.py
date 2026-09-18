"""Does the ground station's second opinion actually beat the payload's?

    .venv/bin/python training/eval_rescore.py --n 300

Cuts 160 px crops centred on labelled people -- exactly what the thermal gate
hands the detector -- and scores each one twice:

    payload path   yolov8n @ 160 px   (what runs on the Pi)
    ground path    yolov8m @ 640 px   (what runs here)

and reports detection rate at a matched confidence threshold, plus the
false-positive rate on crops containing no person at all. Both halves matter:
a second opinion that finds more people but also invents more of them has not
helped anyone.

The crops come from RGBTDronePerson's visible imagery, which was captured at
night and is the reason the RGB fine-tune in this project failed. That makes it
a hard test rather than a flattering one -- if the larger model helps here it
helps in the conditions a search actually runs in.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
CROP = 160


def person_boxes(label_path: pathlib.Path, w: int, h: int) -> list[tuple[float, float]]:
    """YOLO-format centres for class 0, in pixels."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "0":
            out.append((float(parts[1]) * w, float(parts[2]) * h))
    return out


def cut(img: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    h, w = img.shape[:2]
    x0 = int(np.clip(cx - CROP / 2, 0, max(w - CROP, 0)))
    y0 = int(np.clip(cy - CROP / 2, 0, max(h - CROP, 0)))
    patch = img[y0:y0 + CROP, x0:x0 + CROP]
    return patch if patch.shape[:2] == (CROP, CROP) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(REPO / "data/converted/rgbt/images/val"))
    ap.add_argument("--labels", default=str(REPO / "data/converted/rgbt/labels/val"))
    ap.add_argument("--n", type=int, default=300, help="positive crops to test")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="matched decision threshold for both models")
    ap.add_argument("--out", default=str(REPO / "results/detector/rescore_ab.json"))
    args = ap.parse_args()

    from saresq.rescore.engine import Rescorer

    random.seed(0)
    files = sorted(glob.glob(str(pathlib.Path(args.images) / "*.jpg")))
    random.shuffle(files)

    pos, neg = [], []
    for f in files:
        if len(pos) >= args.n and len(neg) >= args.n:
            break
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        lab = pathlib.Path(args.labels) / (pathlib.Path(f).stem + ".txt")
        centres = person_boxes(lab, w, h)
        for cx, cy in centres:
            if len(pos) < args.n:
                c = cut(img, cx, cy)
                if c is not None:
                    pos.append(c)
        # A negative crop: a random patch at least 120 px from every person.
        if len(neg) < args.n:
            for _ in range(6):
                rx, ry = random.uniform(CROP, w - CROP), random.uniform(CROP, h - CROP)
                if all((rx - cx) ** 2 + (ry - cy) ** 2 > 120 ** 2 for cx, cy in centres):
                    c = cut(img, rx, ry)
                    if c is not None:
                        neg.append(c)
                    break
    print(f"crops: {len(pos)} with a person, {len(neg)} without")

    # A 2x2 rather than a single comparison, because the two deployment paths
    # differ in BOTH model size and input size, and "is it the model or the
    # resolution?" is the first question anyone should ask of the result.
    arms = {
        "payload  yolov8n @160": Rescorer("yolov8n.pt", imgsz=160, conf=0.01),
        "         yolov8n @640": Rescorer("yolov8n.pt", imgsz=640, conf=0.01),
        "         yolov8m @160": Rescorer("yolov8m.pt", imgsz=160, conf=0.01),
        "ground   yolov8m @640": Rescorer("yolov8m.pt", imgsz=640, conf=0.01),
    }
    report: dict = {"n_pos": len(pos), "n_neg": len(neg), "conf": args.conf, "arms": {}}

    for name, eng in arms.items():
        if not eng.available:
            print(f"{name}: unavailable -- {eng.why}")
            continue
        p_pos, p_neg = [], []
        for i in range(0, len(pos), 16):
            p_pos += [r.p for r in eng.score(pos[i:i + 16])]
        for i in range(0, len(neg), 16):
            p_neg += [r.p for r in eng.score(neg[i:i + 16])]
        hit = sum(1 for p in p_pos if p >= args.conf) / max(len(p_pos), 1)
        fp = sum(1 for p in p_neg if p >= args.conf) / max(len(p_neg), 1)
        report["arms"][name] = {
            "detection_rate": round(hit, 4),
            "false_positive_rate": round(fp, 4),
            "mean_p_on_person": round(float(np.mean(p_pos)) if p_pos else 0.0, 4),
            "params": eng.params,
        }
        print(f"{name}:  detect {hit:.3f}   false-pos {fp:.3f}   "
              f"mean p {np.mean(p_pos) if p_pos else 0:.3f}")

    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
