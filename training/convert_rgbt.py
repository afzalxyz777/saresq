"""Convert RGBTDronePerson (Pascal-VOC XML, verified against the real download)
into YOLO format for both the thermal and visible images (Section 7.5 /
Section 11.1).

Verified structure (checked against the actual downloaded zip, not
assumed): <zip>/{train,val}/{annotation,thermal,visible}/NNNNN.{xml,jpg}.
Annotation classes actually present: person, rider, crowd. 640x512 for
both modalities, already pixel-registered (same resolution, no separate
warp needed to overlay one on the other).

    python training/convert_rgbt.py --zip data/raw/rgbtdroneperson/RGBTDronePerson/RGBTDronePerson.zip \
        --out data/converted/rgbt

Writes:
  <out>/labels/{train,val}/NNNNN.txt        (YOLO format, class 0 = person)
  <out>/images_thermal/{train,val}/NNNNN.jpg
  <out>/images_visible/{train,val}/NNNNN.jpg
  <out>/rgbt.yaml                            (Ultralytics dataset config, visible images)
  <out>/verify/*.jpg                         (20 random images with boxes drawn -- LOOK AT THESE)
"""
from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np

KEEP_CLASSES = {"person": 0, "rider": 0}  # both -> class 0; "crowd" is dropped (a region label, not an object)
SPLITS = ["train", "val"]


def parse_annotation(xml_bytes: bytes) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in KEEP_CLASSES:
            continue
        bb = obj.find("bndbox")
        xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
        xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)
        cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        boxes.append((KEEP_CLASSES[name], cx, cy, bw, bh))
    return w, h, boxes


def convert(zip_path: str, out_dir: str, seed: int = 0) -> dict[str, int]:
    out = Path(out_dir)
    counts = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for split in SPLITS:
            ann_files = sorted(n for n in names if n.startswith(f"{split}/annotation/") and n.endswith(".xml"))
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            (out / "images_thermal" / split).mkdir(parents=True, exist_ok=True)
            (out / "images_visible" / split).mkdir(parents=True, exist_ok=True)

            n_written = 0
            for ann_path in ann_files:
                stem = Path(ann_path).stem
                w, h, boxes = parse_annotation(zf.read(ann_path))

                label_path = out / "labels" / split / f"{stem}.txt"
                label_path.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for c, cx, cy, bw, bh in boxes))

                for modality in ("thermal", "visible"):
                    src = f"{split}/{modality}/{stem}.jpg"
                    if src in names:
                        dst = out / f"images_{modality}" / split / f"{stem}.jpg"
                        dst.write_bytes(zf.read(src))
                n_written += 1
            counts[split] = n_written

        _write_dataset_yaml(out)
        _draw_verification_samples(zf, out, seed)
    return counts


def _write_dataset_yaml(out: Path) -> None:
    yaml_text = f"""# Ultralytics dataset config for RGBTDronePerson (visible images, RGB fine-tuning stage)
path: {out.resolve()}
train: images_visible/train
val: images_visible/val
names:
  0: person
"""
    (out / "rgbt.yaml").write_text(yaml_text)


def _draw_verification_samples(zf: zipfile.ZipFile, out: Path, seed: int, n: int = 20) -> None:
    """Section 7.5's own warning: 'verify the conversion visually... this is
    where projects lose days.' Draws boxes on both modalities for n random
    images so a mislabeled class or a wrong xmin/ymin axis is obvious."""
    verify_dir = out / "verify"
    verify_dir.mkdir(exist_ok=True)
    rng = random.Random(seed)
    label_files = list((out / "labels" / "train").glob("*.txt")) + list((out / "labels" / "val").glob("*.txt"))
    sample = rng.sample(label_files, min(n, len(label_files)))

    for label_path in sample:
        split = label_path.parent.name
        stem = label_path.stem
        for modality in ("thermal", "visible"):
            img_path = out / f"images_{modality}" / split / f"{stem}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            for line in label_path.read_text().strip().splitlines():
                _, cx, cy, bw, bh = map(float, line.split())
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(str(verify_dir / f"{split}_{stem}_{modality}.jpg"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="data/converted/rgbt")
    args = ap.parse_args()
    counts = convert(args.zip, args.out)
    print(f"converted: {counts}")
    print(f"wrote verification images to {args.out}/verify -- look at them before trusting this converter")


if __name__ == "__main__":
    main()
