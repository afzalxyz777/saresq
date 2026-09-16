"""Score a thermal detector on each source dataset SEPARATELY.

    python training/eval_per_source.py results/detector/v8n_p3_thermalmix2_640/weights/best.pt

WHY THIS EXISTS
``thermal_combined`` pools two datasets whose validation splits are wildly
unequal -- 290 HIT-UAV frames against 1225 RGBTDronePerson frames -- so the
pooled Person AP50 that ``train_detector.py`` prints is ~81% a score on the
harder urban/night set. It moved from 0.4947 to 0.5216 between the 20- and
45-epoch checkpoints, which reads as a small gain; per source, the same two
checkpoints went 0.8092 -> 0.8499 on HIT-UAV and 0.5231 -> 0.5443 on RGBT.
The pooled number is not wrong, it is unreadable, and the combined dataset's
own config says so in capitals: EVALUATE PER SOURCE, NOT POOLED.

The pooled number is also inflated by OtherVehicle, a junk catch-all that
swung 0.049 -> 0.439 between those checkpoints while Person barely moved.
Reporting mAP50 from a combined run would credit the model for learning to
suppress a class nobody acts on.

The decision rule this script serves: a combined model is adopted only if it
HOLDS HIT-UAV Person AP50 while raising RGBT Person AP50. A gain on the mean
that costs HIT-UAV is a regression, because HIT-UAV is the nadir-aerial
domain the payload actually flies.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent

# RGBT's own yaml declares nc=1 ('person'). A 4-class model validated against
# it would have its class indices reinterpreted, so it gets a 4-class view of
# the same files here. Only class 0 is populated; Ultralytics reports AP for
# the classes present and omits the rest, which is the honest outcome -- the
# alternative is inventing a 0.0 for a class the split never contained.
NAMES4 = {0: "Person", 1: "Car", 2: "Bicycle", 3: "OtherVehicle"}

SOURCES = {
    "hituav": REPO / "data/converted/hituav",
    "rgbt_thermal": REPO / "data/converted/rgbt_thermal",
}


def _yaml_for(root: pathlib.Path) -> str:
    body = [f"path: {root}", "train: images/train", "val: images/val", "names:"]
    body += [f"  {i}: {n}" for i, n in NAMES4.items()]
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fh.write("\n".join(body) + "\n")
    fh.close()
    return fh.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None, help="write JSON here (default: beside the weights)")
    args = ap.parse_args()

    from ultralytics import YOLO

    w = pathlib.Path(args.weights).resolve()
    model = YOLO(str(w))

    out: dict = {"weights": str(w), "imgsz": args.imgsz, "per_source": {}}
    for name, root in SOURCES.items():
        if not (root / "images/val").is_dir():
            continue
        m = model.val(data=_yaml_for(root), imgsz=args.imgsz, device=args.device,
                      split="val", plots=False, verbose=False)
        # ap_class_index maps a row of box.ap50 back to a class id; Person is 0.
        idx = list(m.ap_class_index)
        person = float(m.box.ap50[idx.index(0)]) if 0 in idx else None
        out["per_source"][name] = {
            # Count image files only: a cache=disk run leaves a sibling .npy
            # next to every .jpg, which doubled this number the first time.
            "images": sum(1 for f in (root / "images/val").iterdir()
                          if f.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            "person_AP50": person,
            "person_AP50_95": float(m.box.ap[idx.index(0)]) if 0 in idx else None,
            "all_classes_seen": {NAMES4[c]: float(m.box.ap50[i]) for i, c in enumerate(idx)},
        }

    ps = out["per_source"]
    scores = [v["person_AP50"] for v in ps.values() if v["person_AP50"] is not None]
    out["person_AP50_mean"] = sum(scores) / len(scores) if scores else None

    dest = pathlib.Path(args.out) if args.out else w.parent.parent / "per_source.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"\n{'source':<16}{'imgs':>6}{'Person AP50':>14}{'AP50-95':>10}")
    for name, v in ps.items():
        print(f"{name:<16}{v['images']:>6}{v['person_AP50']:>14.4f}{v['person_AP50_95']:>10.4f}")
    print(f"{'mean':<16}{'':>6}{out['person_AP50_mean']:>14.4f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
