"""Build the labelled 18-feature fusion dataset from RGBTDronePerson (Section 10.6).

The fusion head needs (feature vector, is_survivor) pairs, and until real
flights exist there is exactly one public source of *paired, registered*
visible+thermal aerial imagery of people: RGBTDronePerson. This script runs
the real pipeline over it -- the actual gate, the actual blob extractor, the
actual detector, the actual feature assembler -- and labels each candidate
against the dataset's ground truth.

    python training/make_fusion_dataset.py \
        --weights results/detector/v8n_p2_rgbt_640/weights/best.pt \
        --split val --out results/fusion/fusion_dataset.npz

Three things about this dataset that must not be quietly forgotten, because
each one bounds what the trained head can honestly claim:

**1. It is built from the detector's VALIDATION split, never its training
split.** The p_rgb feature is a detector score. If it were measured on images
the detector had memorised, the fusion head would learn to trust p_rgb far
more than it deserves, and the failure would only surface in flight. This is
the single most important line in the file.

**2. Three of the eighteen features cannot be measured here.** ``hits`` and
``hit_ratio`` are properties of a track across frames; ``alt_band`` is a
property of a mission pass. A still-image dataset has none of those, so this
script emits a constant for them and ``train_fusion.py`` pins their weights
to zero rather than letting the optimiser fit noise. They get calibrated
from operator verdicts (``training/export_verdicts.py``) once real flights
produce them. A head that "learned" a track weight from constant input would
be actively misleading.

**3. The thermal frames are simulated, not measured.** RGBTDronePerson's
thermal camera is not an MLX90640, so ``thermal_degradation.py`` matches it
by ground sample distance (a person lands at ~2 px) rather than by
resolution. The noise level there is an explicit estimate, flagged in that
module, pending a real side-by-side comparison.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from saresq.fuse.features import (  # noqa: E402
    FEATURE_NAMES, ContextEvidence, RegistrationEvidence, RgbEvidence,
    ThermalEvidence, TrackEvidence, assemble_features,
)
from saresq.gate.blobs import extract_blobs  # noqa: E402
from saresq.gate.contrast import compute_contrast, gate_mask  # noqa: E402
from training.thermal_degradation import TARGET_PERSON_PX, degrade_thermal  # noqa: E402

# --- Unit bridge: 8-bit grey levels vs. radiometric kelvin -------------------
# The gate is unit-free in z, but its two floors (sigma_min_k, dt_min_k) are
# stated in kelvin because the MLX90640 reports kelvin. A public 8-bit thermal
# image has no radiometric scale at all, so those floors have to be restated in
# grey levels or they silently stop doing their job: 0.5 K read as "0.5 grey
# levels" is no floor whatsoever, and the gate would then fire on pure noise in
# a flat scene.
#
# 20 K across the full 0-255 range is a stated ASSUMPTION about how these
# images were normalised, not a measured calibration. It is exposed as a CLI
# flag so the sensitivity of everything downstream to it can be tested.
DEFAULT_GREY_PER_KELVIN = 255.0 / 20.0

# Track and altitude features are structurally unavailable in a still-image
# dataset (see note 2 above). Constants, clearly named, never zero-by-accident.
STILL_IMAGE_HITS = 1
STILL_IMAGE_AGE = 1
STILL_IMAGE_ALT_BAND = 0
UNAVAILABLE_FEATURES = ("hits", "hit_ratio", "alt_band")


def load_yolo_labels(path: pathlib.Path, w: int, h: int) -> np.ndarray:
    """YOLO-normalised label file -> (N, 4) xyxy pixel boxes for class 0."""
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split()[:5])
        if int(cls) != 0:
            continue
        rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy -> (N,M) IoU."""
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


def degradation_scale(gt_boxes: np.ndarray, fallback: float) -> float:
    """Downsample factor that puts a typical person at TARGET_PERSON_PX.

    Uses the median box in this frame rather than the mean: a single mislabelled
    or heavily-occluded box should not set the scale for the whole image.
    """
    if len(gt_boxes) == 0:
        return fallback
    sizes = np.concatenate([gt_boxes[:, 2] - gt_boxes[:, 0], gt_boxes[:, 3] - gt_boxes[:, 1]])
    median_px = float(np.median(sizes))
    if median_px <= 0:
        return fallback
    return min(1.0, TARGET_PERSON_PX / median_px)


def crop_window(cx: float, cy: float, size: int, w: int, h: int) -> tuple[int, int, int, int]:
    """A ``size``x``size`` window centred on (cx, cy), shifted to stay in frame.

    Shifted rather than clipped: the detector's input tensor is a fixed square,
    so a clipped window would have to be padded, and padding a crop that sits
    near the frame edge puts a hard synthetic border right next to the target.
    Sliding the window keeps every pixel real.
    """
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x1 = max(0, min(x1, w - size)) if w >= size else 0
    y1 = max(0, min(y1, h - size)) if h >= size else 0
    return x1, y1, x1 + min(size, w), y1 + min(size, h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--data", default=str(repo / "data" / "converted" / "rgbt" / "rgbt.yaml"))
    ap.add_argument("--split", default="val", help="MUST be the detector's val split (see module docstring)")
    ap.add_argument("--weights", required=True, help="RGB-domain detector for Pass B")
    ap.add_argument("--hazard", default=None, help="optional TFLite hazard model for Pass E context")
    ap.add_argument("--out", default=str(repo / "results" / "fusion" / "fusion_dataset.npz"))
    ap.add_argument("--limit", type=int, default=0, help="cap images, for a smoke test")
    ap.add_argument("--crop-px", type=int, default=160)
    ap.add_argument("--crop-large-px", type=int, default=224)
    ap.add_argument("--large-blob-px", type=int, default=9)
    ap.add_argument("--max-blobs", type=int, default=12, help="per image, ranked; runtime budget is 2-6")
    ap.add_argument("--conf", type=float, default=0.05, help="low: we want the score, not a decision")
    ap.add_argument("--z-t", type=float, default=2.5)
    ap.add_argument("--grey-per-kelvin", type=float, default=DEFAULT_GREY_PER_KELVIN)
    ap.add_argument("--match-iou", type=float, default=0.10)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from ultralytics import YOLO
    from training.train_detector import pick_device

    device = pick_device(args.device)
    detector = YOLO(args.weights)

    hazard = None
    if args.hazard:
        from saresq.detect.hazard import HazardClassifier
        hazard = HazardClassifier(args.hazard)

    cfg = yaml.safe_load(open(args.data))
    root = pathlib.Path(cfg["path"])
    vis_dir = root / cfg[args.split]
    # Addressed explicitly, not by substituting into the visible path. The
    # visible folder has to be named exactly "images" for Ultralytics' label
    # lookup to work, and "images" is a substring of "images_thermal", so a
    # replace() here would silently produce nonsense.
    thr_dir = root / "images_thermal" / args.split
    lbl_dir = root / "labels" / args.split
    for name, d in (("visible", vis_dir), ("thermal", thr_dir), ("labels", lbl_dir)):
        if not d.is_dir():
            raise SystemExit(f"{name} directory missing: {d}")

    images = sorted(vis_dir.glob("*.jpg")) + sorted(vis_dir.glob("*.png"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images under {vis_dir}")

    # Gate floors, restated in this dataset's units (see the unit-bridge note).
    sigma_min = 0.15 * args.grey_per_kelvin
    dt_min = 0.5 * args.grey_per_kelvin

    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []  # image index, so the split never straddles one frame
    n_blobs = n_gt = 0
    # Distinct ground-truth people the gate put a candidate on. Counting
    # *candidates* instead would double-count: two blobs on one person would
    # report 200% recall, and this number feeds eval_sweep_width.py, which sets
    # the search model's central constant.
    n_gt_found = 0

    for img_idx, vis_path in enumerate(images):
        thr_path = thr_dir / vis_path.name
        vis = cv2.imread(str(vis_path))
        thr = cv2.imread(str(thr_path), cv2.IMREAD_GRAYSCALE)
        if vis is None or thr is None:
            continue
        h, w = vis.shape[:2]
        gt = load_yolo_labels(lbl_dir / (vis_path.stem + ".txt"), w, h)
        n_gt += len(gt)

        scale = degradation_scale(gt, fallback=TARGET_PERSON_PX / 12.0)
        small = degrade_thermal(thr.astype(np.float32), scale)
        t_bg, sigma, z, dT = compute_contrast(small, sigma_min_k=sigma_min)
        mask = gate_mask(z, dT, z_t=args.z_t, dt_min_k=dt_min, two_sided=True)
        blobs = extract_blobs(z, dT, mask)
        if not blobs:
            continue
        blobs = sorted(blobs, key=lambda b: -b.rank_score)[: args.max_blobs]

        # Pass E context is a property of the SCENE, so it is computed once on
        # the whole frame and shared by every candidate in it -- not per crop.
        # AIDER trained this classifier on entire aerial scenes; a 160 px crop
        # is out of its domain, and it would answer anyway.
        p_flood = p_fire = p_collapse = 0.0
        if hazard is not None:
            p_flood, p_fire, p_collapse = hazard.hazard_features(vis)

        # Blob bbox (degraded-thermal px) -> full-resolution visible px.
        # The two modalities are pixel-registered at 640x512 and share one
        # label file, so this is a pure scale change with no homography.
        matched_gt: set[int] = set()
        crops, metas = [], []
        for blob in blobs:
            u0, v0, u1, v1 = blob.bbox_t
            bx1, by1 = u0 / scale, v0 / scale
            bx2, by2 = (u1 + 1) / scale, (v1 + 1) / scale
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            size = args.crop_large_px if blob.area_t >= args.large_blob_px else args.crop_px
            x1, y1, x2, y2 = crop_window(cx, cy, size, w, h)
            crops.append(vis[y1:y2, x1:x2])
            metas.append((blob, np.array([[bx1, by1, bx2, by2]], dtype=np.float32), (x1, y1)))

        results = detector.predict(crops, conf=args.conf, verbose=False, device=device)

        for (blob, thermal_box, (ox, oy)), result, crop in zip(metas, results, crops):
            n_blobs += 1
            # --- Pass B: RGB evidence -------------------------------------
            rgb_ev = None
            det_box_global = None
            if result.boxes is not None and len(result.boxes):
                scores = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                best = int(np.argmax(scores))
                bw = float(boxes[best, 2] - boxes[best, 0])
                bh = float(boxes[best, 3] - boxes[best, 1])
                rgb_ev = RgbEvidence(
                    score=float(scores[best]),
                    box_area_px=bw * bh,
                    crop_area_px=float(crop.shape[0] * crop.shape[1]),
                )
                det_box_global = boxes[best:best + 1] + np.array([ox, oy, ox, oy], dtype=np.float32)

            # --- Pass C: registration agreement ---------------------------
            reg_ev = None
            if det_box_global is not None:
                overlap = float(iou_matrix(thermal_box, det_box_global)[0, 0])
                tc = thermal_box[0, :2] + (thermal_box[0, 2:] - thermal_box[0, :2]) / 2
                dc = det_box_global[0, :2] + (det_box_global[0, 2:] - det_box_global[0, :2]) / 2
                diag = float(np.hypot(crop.shape[1], crop.shape[0]))
                reg_ev = RegistrationEvidence(
                    iou=overlap,
                    centroid_dist_norm=float(np.linalg.norm(tc - dc) / diag),
                )

            features = assemble_features(
                thermal=ThermalEvidence(
                    z_peak=abs(float(blob.z_peak)), dT_peak=float(blob.dT_peak),
                    sign=int(blob.sign), area_t=int(blob.area_t), ecc=float(blob.ecc),
                ),
                rgb=rgb_ev,
                registration=reg_ev,
                track=TrackEvidence(hits=STILL_IMAGE_HITS, age=STILL_IMAGE_AGE),
                context=ContextEvidence(
                    t_bg=float(t_bg) / 255.0,
                    lum=float(crop.mean()) / 255.0,
                    p_flood=p_flood, p_fire=p_fire, p_collapse=p_collapse,
                    alt_band=STILL_IMAGE_ALT_BAND,
                ),
            )

            # --- Label ----------------------------------------------------
            # Matched against the THERMAL blob's footprint, not the detector's
            # box. The label must answer "was there a person where the gate
            # fired?", which is independent of whether the detector found them
            # -- otherwise every gate hit the detector missed would be labelled
            # negative and the head could never learn to recover one.
            label = 0
            if len(gt):
                ious = iou_matrix(thermal_box, gt)[0]
                hit = ious >= args.match_iou
                if not hit.any():
                    # A 2 px thermal blob upscaled back to visible coordinates
                    # is a coarse footprint, so IoU alone is harsh. Falling back
                    # to "the blob's centre lies inside a person's box" recovers
                    # the genuine hits that a strict overlap test would discard.
                    c = thermal_box[0, :2] + (thermal_box[0, 2:] - thermal_box[0, :2]) / 2
                    hit = ((gt[:, 0] <= c[0]) & (c[0] <= gt[:, 2])
                           & (gt[:, 1] <= c[1]) & (c[1] <= gt[:, 3]))
                if hit.any():
                    label = 1
                    matched_gt.update(np.flatnonzero(hit).tolist())

            X.append(features)
            y.append(label)
            groups.append(img_idx)

        n_gt_found += len(matched_gt)

        if (img_idx + 1) % 100 == 0:
            pos = int(np.sum(y))
            print(f"{img_idx + 1}/{len(images)} images | {len(y)} candidates | {pos} positive "
                  f"({100 * pos / max(len(y), 1):.1f}%)", flush=True)

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    groups_arr = np.asarray(groups, dtype=np.int64)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, X=X_arr, y=y_arr, groups=groups_arr,
        feature_names=np.array(FEATURE_NAMES),
        unavailable=np.array(UNAVAILABLE_FEATURES),
    )

    meta = {
        "images": len(images),
        "gt_boxes": n_gt,
        "candidates": int(len(y_arr)),
        "positives": int(y_arr.sum()),
        "base_rate": float(y_arr.mean()) if len(y_arr) else 0.0,
        # Distinct people the gate found / people present. This is the thermal
        # gate's recall on simulated MLX90640 imagery, and it is consumed
        # directly by training/eval_sweep_width.py --gate-recall-json.
        "gate_recall": float(n_gt_found / n_gt) if n_gt else 0.0,
        "gt_found": int(n_gt_found),
        "split": args.split,
        "weights": args.weights,
        "hazard": args.hazard,
        "grey_per_kelvin": args.grey_per_kelvin,
        "unavailable_features": list(UNAVAILABLE_FEATURES),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
