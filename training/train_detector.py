"""Train the survivor detector: P3-P5 baseline vs. P2-P5 variant (Sections 7.4, 7.6).

    # baseline, then the small-object variant
    python training/train_detector.py --variant p3 --epochs 100
    python training/train_detector.py --variant p2 --epochs 100

    # RGB-domain fine-tune, starting from the thermal-pretrained weights
    python training/train_detector.py --variant p2 --data data/converted/rgbt/rgbt.yaml \
        --weights results/detector/v8n_p2_hituav_640/weights/best.pt --epochs 40 --tag rgbt

Why both variants are trained rather than just picking one, stated in the
pixel sizes this payload actually produces (GSD = altitude / focal_px, with
focal_px = 1359.3 and survey altitude 20 m from ``configs/pipeline.yaml``,
giving 14.7 mm/px):

===========================  ===================  =====================
path                         prone adult          standing adult
===========================  ===================  =====================
thermal 32x24 (the gate)     2.6 px               ~0.7 px
RGB 160 px crop              115 x 31 px          31 x 24 px
RGB 640 full-frame fallback  45 x 12 px           12 x 9 px
===========================  ===================  =====================

So the small-object problem is **not** in the crop path -- a 31 px object on a
stride-8 grid is four cells across and the baseline head handles it. It is in
the **full-frame fallback**, the mode the pipeline drops to when the gate has
been starved for three consecutive frames (``gate.fallback`` in the config).
There a standing person is 9-12 px, which is one to one-and-a-half stride-8
cells, and the P2 head's stride-4 level is the only change that puts more than
a single cell on the target. HIT-UAV's own imagery is flown at 60-130 m and
sits in the same regime, which is what makes it the right dataset to measure
the difference on.

That is the claim ``eval_pixels_on_target.py`` exists to test, so the two runs
must differ in nothing but the head.

Class policy: HIT-UAV's vehicle classes (Car, Bicycle, OtherVehicle) are kept,
not collapsed to person-only. They are the training signal for *rejecting* the
warm car bonnet, which is the single most common thermal false positive in a
disaster zone. Person AP is the headline metric; the vehicle classes are there
to make it honest.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "data" / "converted" / "hituav" / "hituav.yaml"


def _stem_for(base: str | None) -> str:
    """Run-name stem from the base checkpoint: yolo11n.pt -> 'v11n'."""
    if not base:
        return "v8n"
    b = pathlib.Path(base).stem.lower()
    if b.startswith("yolo11"):
        return "v11" + b[len("yolo11"):]
    if b.startswith("yolov"):
        return "v" + b[len("yolov"):]
    return b


def pick_device(requested: str | None) -> str:
    """Resolve 'auto' to the best backend actually present."""
    if requested and requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(variant: str, weights: str | None, base: str | None = None):
    """P3 loads the stock checkpoint; P2 grafts those weights onto a 4-level head.

    ``.load()`` copies every tensor whose name and shape match, so the backbone
    and neck transfer intact and only the new stride-4 branch starts cold. That
    is what keeps the comparison fair: both variants inherit the same COCO
    features, and the difference measured is the head, not the initialisation.
    """
    from ultralytics import YOLO

    if weights:
        return YOLO(weights)
    if base:
        # An explicit base checkpoint (yolo11n.pt, yolov8s.pt, ...). Verified
        # 2026-09-16 that yolo11n emits the same (1, 4+nc, n_anchors) tensor as
        # yolov8n -- DFL baked in, no NMS -- so saresq/detect/decode.py needs no
        # change and the A/B is a pure backbone swap.
        if variant == "p2":
            raise SystemExit("--model with --variant p2 is not supported: the "
                             "P2 graft needs a matching *-p2.yaml, not a .pt")
        return YOLO(base)
    if variant == "p3":
        return YOLO("yolov8n.pt")
    if variant == "p2":
        # The 'n' in the stem is load-bearing: Ultralytics reads the scale
        # letter out of the filename. Passing the unscaled 'yolov8-p2.yaml'
        # only *warns* and assumes nano, which would silently invalidate the
        # comparison the moment that default changed.
        return YOLO("yolov8n-p2.yaml").load("yolov8n.pt")
    raise ValueError(f"unknown variant {variant!r}, expected 'p3' or 'p2'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["p3", "p2"], required=True)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--weights", default=None, help="start from a checkpoint (fine-tuning)")
    ap.add_argument("--model", default=None,
                    help="base checkpoint for a fresh run, e.g. yolo11n.pt (p3 only)")
    ap.add_argument("--lr0", type=float, default=0.001,
                    help="0.001 from cold; use ~0.0003 when continuing a converged run, "
                         "or the warm restart throws away several epochs recovering")
    ap.add_argument("--warmup", type=float, default=3.0,
                    help="warmup epochs; drop to 1 when continuing a trained checkpoint")
    ap.add_argument("--stem", default=None,
                    help="override the run-name stem (default: derived from --model)")
    ap.add_argument("--tag", default="hituav", help="dataset tag used in the run name")
    ap.add_argument("--imgsz", type=int, default=640, help="native HIT-UAV size; do not upscale")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--fraction", type=float, default=1.0, help="<1.0 for a smoke test")
    ap.add_argument("--cache", default="False", help="False | disk | ram (ram needs ~2 GB for HIT-UAV)")
    ap.add_argument("--amp", default="True", choices=["True", "False"],
                    help="MUST be False for the P2 head on MPS -- see the note in main()")
    ap.add_argument("--out", default=str(REPO / "results" / "detector"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    # The stem names the ARCHITECTURE, not the dataset, so a YOLO11n A/B lands
    # in its own directory instead of overwriting the v8n numbers the deck quotes.
    stem = args.stem or _stem_for(args.model)
    name = f"{stem}_{args.variant}_{args.tag}_{args.imgsz}"
    out_root = pathlib.Path(args.out)

    model = build_model(args.variant, args.weights, args.model)
    info = model.info()  # (layers, params, gradients, GFLOPs)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=device,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        fraction=args.fraction,
        cache={"False": False, "True": True}.get(args.cache, args.cache),
        # AMP off for P2, and this is not a tuning preference -- it is a
        # correctness fix. The stride-4 level takes the anchor count from 8,400
        # to 34,000 at 640 px, and the task-aligned assigner's score tensor
        # scales with it. Under fp16 on MPS that overflowed: measured
        # 2026-09-10, cls_loss ran 4.97 -> 46 -> **-1224** by iteration 6, a
        # classification loss that cannot be negative. It does not raise; it
        # just trains to garbage. Verified separately that the same run in
        # fp32 stays finite.
        amp={"False": False, "True": True}[args.amp],
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        warmup_epochs=args.warmup,
        # Augmentation for nadir aerial imagery (Correction C9): when the camera
        # points straight down there is no canonical "up", so any rotation or
        # flip is a physically valid view of the same scene -- unlike a
        # ground-level dataset, where upside-down people are label noise.
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,        # altitude varies across the survey -> aggressive scale jitter
        translate=0.1,
        mosaic=1.0,
        close_mosaic=10,  # last 10 epochs see whole frames, matching inference
        # Thermal imagery carries no hue or saturation, so jittering them would
        # teach the model to be invariant to a channel that does not exist.
        # Brightness stands in for ambient temperature drift, which does.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        project=str(out_root),
        name=name,
        exist_ok=True,
        resume=args.resume,
        save_period=-1,   # best.pt + last.pt only; per-epoch copies cost 6 MB each
        plots=True,
        val=True,
    )

    run_dir = out_root / name
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=device, split="val", plots=False)

    # A machine-readable sidecar so the P2-vs-P3 table, the dashboard and the
    # deck all read the same numbers instead of re-deriving them from stdout.
    per_class = {}
    try:
        for i, cls_name in metrics.names.items():
            per_class[cls_name] = {
                "AP50": float(metrics.box.ap50[list(metrics.ap_class_index).index(i)]),
                "AP50_95": float(metrics.box.ap[list(metrics.ap_class_index).index(i)]),
            }
    except (ValueError, AttributeError, IndexError):
        pass  # a class absent from the val split has no AP; leave it out rather than fake a 0

    summary = {
        "variant": args.variant,
        "base_model": args.model or args.weights or f"yolov8n ({args.variant})",
        "lr0": args.lr0,
        "data": args.data,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "device": device,
        "layers": info[0], "parameters": info[1], "gflops": info[3],
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": per_class,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # readiness.py globs results/detector/**/results.csv -- Ultralytics already
    # writes it there, but copy the sidecar up so the dashboard finds one file
    # per variant without walking into weights/.
    shutil.copy(run_dir / "summary.json", out_root / f"{name}_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
