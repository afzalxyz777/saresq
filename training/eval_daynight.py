"""Does the detector still work in daylight? Score HIT-UAV val split by light.

    python training/eval_daynight.py \
        --weights results/detector/v8n_p3_thermalmix2_640/weights/best.pt

Writes ``results/detector/daynight.json``.

This exists because "thermal is a night sensor" is the most common wrong
intuition about this payload, and the answer should be a measurement rather
than a reassurance. Thermal needs a *temperature difference*, not light --
and daylight works against us by heating the background toward body
temperature, which is the opposite of the way people assume it fails.

HIT-UAV encodes light condition as the first underscore-separated field of
every filename (``<light>_<alt>_<angle>_<n>_<id>.jpg``); its README states the
corpus covers "daylight intensity (day and night)" but never says which digit
is which, so this script does not trust a guess. It infers the mapping from
the physics, using *cars* as the discriminator:

  * In sunlight a parked car's metal roof bakes well above the ground
    around it, so cars read HOTTER than background.
  * Parked overnight, the same car has radiated its heat away while the
    tarmac underneath still holds the day's, so cars read COLDER.

Mean frame brightness cannot do this job -- the camera auto-gains each frame,
so day and night images have nearly identical means (131 vs 127 measured).
Person contrast alone is suggestive but not decisive; the car sign flip is
unambiguous, and the two agree.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parent.parent
PERSON, CAR = "0", "1"          # HIT-UAV class order, see hituav.yaml


def _contrast(images: pathlib.Path, labels: pathlib.Path) -> dict:
    """Mean (object - whole-frame) grey level per light group, per class."""
    acc: dict[str, dict[str, list[float]]] = {}
    for img_p in sorted(images.glob("*.jpg")):
        group = img_p.name.split("_")[0]
        lbl_p = labels / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue
        arr = np.asarray(Image.open(img_p).convert("L"), dtype=np.float32)
        h, w = arr.shape
        bg = float(arr.mean())
        per_class = acc.setdefault(group, {PERSON: [], CAR: []})
        for line in lbl_p.read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5 or parts[0] not in per_class:
                continue
            xc, yc, bw, bh = (float(v) for v in parts[1:])
            x0, x1 = int((xc - bw / 2) * w), int((xc + bw / 2) * w)
            y0, y1 = int((yc - bh / 2) * h), int((yc + bh / 2) * h)
            crop = arr[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]
            if crop.size:
                per_class[parts[0]].append(float(crop.mean()) - bg)
    return {
        g: {k: (statistics.mean(v) if v else float("nan"), len(v))
            for k, v in cls.items()}
        for g, cls in acc.items()
    }


def _label_groups(contrast: dict) -> dict[str, str]:
    """Name each group day/night from the sign of its car contrast."""
    named = {g: ("night" if c[CAR][0] < 0 else "day") for g, c in contrast.items()}
    if sorted(named.values()) != ["day", "night"]:
        raise SystemExit(
            f"cannot separate day from night -- car contrasts {named} do not "
            "straddle zero. Inspect the frames before quoting any number."
        )
    return named


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=str(REPO / "data/converted/hituav"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(REPO / "results/detector/daynight.json"))
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    root = pathlib.Path(args.data_root)
    images = root / "images" / args.split
    labels = root / "labels" / args.split
    if not images.is_dir():
        raise SystemExit(f"no such split: {images}")

    contrast = _contrast(images, labels)
    names = _label_groups(contrast)
    print("light-group inference (grey levels above frame mean):")
    for g, cond in sorted(names.items(), key=lambda kv: kv[1]):
        c = contrast[g]
        print(f"  group {g} -> {cond:<5}  person {c[PERSON][0]:+6.1f} "
              f"(n={c[PERSON][1]})   car {c[CAR][0]:+6.1f} (n={c[CAR][1]})")

    # Ultralytics finds labels by substituting the literal "/images/" in the
    # image path, and resolves symlinks before doing it -- so the per-group
    # splits must be HARD links inside a directory really named "images".
    work = REPO / "results/detector/_daynight_split"
    out: dict[str, dict] = {}
    for group, cond in names.items():
        for sub in ("images", "labels"):
            (work / cond / sub / args.split).mkdir(parents=True, exist_ok=True)
        n = 0
        for img_p in sorted(images.glob(f"{group}_*.jpg")):
            lbl_p = labels / (img_p.stem + ".txt")
            for src, dst in ((img_p, work / cond / "images" / args.split / img_p.name),
                             (lbl_p, work / cond / "labels" / args.split / lbl_p.name)):
                if dst.exists():
                    dst.unlink()
                if src.exists():
                    dst.hardlink_to(src.resolve())
            n += 1
        yaml_p = work / f"{cond}.yaml"
        yaml_p.write_text(
            f"path: {work / cond}\ntrain: images/{args.split}\n"
            f"val: images/{args.split}\n"
            "names:\n  0: Person\n  1: Car\n  2: Bicycle\n  3: OtherVehicle\n")

        res = YOLO(args.weights).val(
            data=str(yaml_p), imgsz=args.imgsz, device=args.device,
            project=str(work), name=f"val_{cond}", exist_ok=True,
            plots=False, verbose=False)
        by_class = {res.names[int(c)]: float(res.box.ap50[i])
                    for i, c in enumerate(res.box.ap_class_index)}
        out[cond] = {"images": n, "person_ap50": by_class.get("Person"),
                     "map50": float(res.box.map50), "ap50_by_class": by_class,
                     "car_contrast": contrast[group][CAR][0],
                     "person_contrast": contrast[group][PERSON][0]}
        print(f"{cond}: {n} images, Person AP50 {out[cond]['person_ap50']:.4f}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
