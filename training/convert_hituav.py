"""Convert HIT-UAV (normal_xml variant) to YOLO format (Section 7.5).

Verified against the real download, not assumed:
  <root>/normal_xml/JPEGImages/<daylight>_<altitude>_<angle>_0_<id>.jpg  (640x512, 1-channel)
  <root>/normal_xml/Annotations/<same-stem>.xml                          (Pascal VOC)
  <root>/normal_xml/ImageSets/Main/{train,val,test,trainval}.txt         (stems, no extension)
Classes actually present: Person, Car, Bicycle, OtherVehicle, DontCare.
Filename altitude range 60-130 m, angle 30-90 deg -- matches the paper.

    python training/convert_hituav.py \
        --root "data/raw/hituav/suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c/normal_xml" \
        --out data/converted/hituav

Writes:
  <out>/images/{train,val,test}/*.jpg
  <out>/labels/{train,val,test}/*.txt
  <out>/meta.csv                 (image, split, altitude_m, angle_deg)
  <out>/hituav.yaml               (Ultralytics dataset config)
  <out>/verify/*.jpg               (20 random images with boxes drawn -- LOOK AT THESE)
"""
from __future__ import annotations

import argparse
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

# Person=0 (the survivor class); Car/Bicycle/OtherVehicle kept as useful hard
# negatives (Section 7.5); DontCare dropped entirely (ambiguous regions, not objects).
CLASS_MAP = {"Person": 0, "Car": 1, "Bicycle": 2, "OtherVehicle": 3}
CLASS_NAMES = ["Person", "Car", "Bicycle", "OtherVehicle"]
SPLITS = ["train", "val", "test"]


def parse_filename_metadata(stem: str) -> tuple[int, int]:
    """`<daylight>_<altitude>_<angle>_0_<id>` -> (altitude_m, angle_deg)."""
    parts = stem.split("_")
    return int(parts[1]), int(parts[2])


def parse_annotation(xml_path: Path) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            continue  # drops DontCare
        bb = obj.find("bndbox")
        xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
        xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)
        cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        boxes.append((CLASS_MAP[name], cx, cy, bw, bh))
    return w, h, boxes


def convert(root_dir: str, out_dir: str, seed: int = 0) -> dict[str, int]:
    root = Path(root_dir)
    out = Path(out_dir)
    counts = {}
    meta_rows = []

    for split in SPLITS:
        split_file = root / "ImageSets" / "Main" / f"{split}.txt"
        stems = [s.strip() for s in split_file.read_text().splitlines() if s.strip()]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        for stem in stems:
            img_src = root / "JPEGImages" / f"{stem}.jpg"
            xml_src = root / "Annotations" / f"{stem}.xml"
            w, h, boxes = parse_annotation(xml_src)

            label_path = out / "labels" / split / f"{stem}.txt"
            label_path.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for c, cx, cy, bw, bh in boxes))
            (out / "images" / split / f"{stem}.jpg").write_bytes(img_src.read_bytes())

            altitude_m, angle_deg = parse_filename_metadata(stem)
            meta_rows.append({"image": stem, "split": split, "altitude_m": altitude_m, "angle_deg": angle_deg,
                               "n_person": sum(1 for c, *_ in boxes if c == 0)})
        counts[split] = len(stems)

    with open(out / "meta.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "altitude_m", "angle_deg", "n_person"])
        writer.writeheader()
        writer.writerows(meta_rows)

    _write_dataset_yaml(out)
    _draw_verification_samples(out, seed)
    return counts


def _write_dataset_yaml(out: Path) -> None:
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_text = f"""# Ultralytics dataset config for HIT-UAV (normal_xml, axis-aligned boxes)
path: {out.resolve()}
train: images/train
val: images/val
test: images/test
names:
{names_block}
"""
    (out / "hituav.yaml").write_text(yaml_text)


def _draw_verification_samples(out: Path, seed: int, n: int = 20) -> None:
    verify_dir = out / "verify"
    verify_dir.mkdir(exist_ok=True)
    rng = random.Random(seed)
    label_files = []
    for split in SPLITS:
        label_files += [(split, p) for p in (out / "labels" / split).glob("*.txt")]
    sample = rng.sample(label_files, min(n, len(label_files)))

    colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 255, 255)}
    for split, label_path in sample:
        stem = label_path.stem
        img_path = out / "images" / split / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for line in label_path.read_text().strip().splitlines():
            cls, cx, cy, bw, bh = line.split()
            cls, cx, cy, bw, bh = int(cls), *map(float, (cx, cy, bw, bh))
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(cls, (255, 255, 255)), 1)
            cv2.putText(img, CLASS_NAMES[cls], (x1, max(0, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors.get(cls, (255, 255, 255)), 1)
        cv2.imwrite(str(verify_dir / f"{split}_{stem}.jpg"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the normal_xml directory")
    ap.add_argument("--out", default="data/converted/hituav")
    args = ap.parse_args()
    counts = convert(args.root, args.out)
    print(f"converted: {counts}")
    print(f"wrote verification images to {args.out}/verify -- look at them before trusting this converter")


if __name__ == "__main__":
    main()
