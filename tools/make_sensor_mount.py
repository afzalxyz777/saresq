"""Generate the thermal + RGB sensor mount as a printable STL.

    ./.venv/bin/python tools/make_sensor_mount.py --out results/mount/

Produces `saresq_sensor_mount.stl` plus a dimensions sheet for the print shop.

## Why this geometry

**Boards are located by their edges, not their screw holes.** Every breakout
vendor puts mounting holes somewhere slightly different, and a hole pattern
that is 1.5 mm out makes a paid-for print into a coaster. Recessed pockets cut
to the PCB outline (+0.4 mm clearance) locate each board positively no matter
where its holes are; screws or a dab of RTV then just stop it lifting.

**Optical baseline is 28 mm.** Derived, not guessed. Parallax error is
baseline/range, and the MLX90640's pixels subtend 30 mrad each
(55 deg / 32 px), so:

    range    parallax at 28 mm    in thermal pixels
     20 m         1.40 mrad             0.047
      8 m         3.50 mrad             0.117
      2 m        14.00 mrad             0.467

against the 0.5 px registration budget in configs/pipeline.yaml. 28 mm stays
inside budget even for a 2 m bench test, which is the tightest case; at survey
altitude it is negligible. Going tighter buys nothing and forces a wall thinner
than 3 mm between the pockets.

**Flat, single-piece, no overhangs.** Prints on any FDM machine lying on its
back with zero supports. The apertures are through-holes, so nothing bridges.
"""
from __future__ import annotations

SUPERSEDED = """
tools/make_sensor_mount.py is SUPERSEDED and must not be run.

The mesh it produces is not printable. Measured on its own output:
not watertight, 105 broken faces, 30 edges shared by more than two faces,
2 zero-area triangles -- the red error faces every CAD viewer showed.

The cause is `trimesh.util.concatenate`, used here to build the slot cutters.
It glues meshes together WITHOUT a boolean union, so each slot handed the
difference operation a self-intersecting solid with interior faces. No
parameter change fixes that; it is what the file does.

Its geometry also had a real design fault this never revealed: the corner M3
holes overran the 3 mm rounded corners by 0.82 mm, so each printed as an open
notch rather than a hole.

Use instead:  ./.venv/bin/python tools/make_payload_cad.py --only mount
That builds the same part as an exact B-rep solid, cut from the SAME pocket and
aperture geometry as the tray, exports STEP as well as STL, and verifies the
STL is watertight and manifold before it will exit 0.

Kept only as a record of what was tried. Delete once the rebuild is printed.
"""
raise SystemExit(SUPERSEDED)

import argparse
import math
import pathlib

import numpy as np
import trimesh

# --- Parameters (mm) --------------------------------------------------------
PLATE_X, PLATE_Y, PLATE_Z = 66.0, 34.0, 3.0
FILLET_R = 3.0

# Thermal breakout: 7SEMI MLX9064X, measured 25 x 21-22 mm.
THERMAL_W, THERMAL_H = 25.0, 22.0
# Raspberry Pi Camera v2: 25 x 23.86 mm.
CAMERA_W, CAMERA_H = 25.0, 24.0

PCB_CLEARANCE = 0.4     # per side; FDM shrinks and boards vary
POCKET_DEPTH = 1.8      # PCB is ~1.6 mm; leaves the board just proud
WALL = 3.0

BASELINE = 28.0         # optical centre to optical centre -- see docstring

THERMAL_APERTURE_D = 11.0   # MLX90640 lens barrel is ~8 mm; generous margin
CAMERA_APERTURE_D = 9.0     # Pi Cam v2 lens barrel ~7.5 mm

MOUNT_HOLE_D = 3.4          # M3 clearance
MOUNT_INSET = 4.5
SLOT_D = 2.4                # M2 clearance
SLOT_LEN = 3.0              # elongation absorbs hole-position error

CABLE_SLOT_W, CABLE_SLOT_H = 18.0, 4.0


def rounded_plate(x: float, y: float, z: float, r: float) -> trimesh.Trimesh:
    """Plate with rounded corners, built as a convex hull of corner cylinders."""
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = trimesh.creation.cylinder(radius=r, height=z, sections=48)
            c.apply_translation([sx * (x / 2 - r), sy * (y / 2 - r), 0])
            parts.append(c)
    return trimesh.util.concatenate(parts).convex_hull


def box_at(w: float, h: float, d: float, cx: float, cy: float, cz: float) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=(w, h, d))
    b.apply_translation([cx, cy, cz])
    return b


def hole(d: float, depth: float, cx: float, cy: float, cz: float) -> trimesh.Trimesh:
    c = trimesh.creation.cylinder(radius=d / 2, height=depth, sections=40)
    c.apply_translation([cx, cy, cz])
    return c


def slot(d: float, length: float, depth: float, cx: float, cy: float, cz: float) -> trimesh.Trimesh:
    """Capsule-shaped slot: elongated along X so screw positions can vary."""
    parts = [hole(d, depth, cx - length / 2, cy, cz), hole(d, depth, cx + length / 2, cy, cz),
             box_at(length, d, depth, cx, cy, cz)]
    return trimesh.util.concatenate(parts)


def build() -> tuple[trimesh.Trimesh, dict]:
    plate = rounded_plate(PLATE_X, PLATE_Y, PLATE_Z, FILLET_R)

    tx, cx = -BASELINE / 2, BASELINE / 2          # pocket centres
    tw, th = THERMAL_W + 2 * PCB_CLEARANCE, THERMAL_H + 2 * PCB_CLEARANCE
    cw, ch = CAMERA_W + 2 * PCB_CLEARANCE, CAMERA_H + 2 * PCB_CLEARANCE

    cuts = []
    # Pockets, opening upward (boards drop in from the top).
    pocket_z = PLATE_Z / 2 - POCKET_DEPTH / 2 + 0.001
    cuts.append(box_at(tw, th, POCKET_DEPTH, tx, 0, pocket_z))
    cuts.append(box_at(cw, ch, POCKET_DEPTH, cx, 0, pocket_z))

    # Lens apertures, straight through.
    cuts.append(hole(THERMAL_APERTURE_D, PLATE_Z * 3, tx, 0, 0))
    cuts.append(hole(CAMERA_APERTURE_D, PLATE_Z * 3, cx, 0, 0))

    # Slotted board screws: two per board, on the long axis, outside the aperture.
    for px, pw, ph in ((tx, tw, th), (cx, cw, ch)):
        for sy in (-1, 1):
            cuts.append(slot(SLOT_D, SLOT_LEN, PLATE_Z * 3, px, sy * (ph / 2 - 2.6), 0))

    # Airframe mounting holes at the four corners.
    for sx in (-1, 1):
        for sy in (-1, 1):
            cuts.append(hole(MOUNT_HOLE_D, PLATE_Z * 3,
                             sx * (PLATE_X / 2 - MOUNT_INSET), sy * (PLATE_Y / 2 - MOUNT_INSET), 0))

    # Ribbon-cable relief on the camera side, so the FFC exits without strain.
    cuts.append(box_at(CABLE_SLOT_W, CABLE_SLOT_H, PLATE_Z * 3, cx, PLATE_Y / 2 - CABLE_SLOT_H / 2 + 0.5, 0))

    body = trimesh.boolean.difference([plate] + [trimesh.util.concatenate([c]) for c in cuts])
    if isinstance(body, list):
        body = body[0]

    t_ifov = math.radians(55.0) / 32
    stats = {
        "plate_mm": [PLATE_X, PLATE_Y, PLATE_Z],
        "baseline_mm": BASELINE,
        "thermal_pocket_mm": [round(tw, 2), round(th, 2), POCKET_DEPTH],
        "camera_pocket_mm": [round(cw, 2), round(ch, 2), POCKET_DEPTH],
        "watertight": bool(body.is_watertight),
        "volume_cm3": round(float(body.volume) / 1000.0, 2),
        "parallax_thermal_px": {f"{r:g} m": round((BASELINE / 1000.0 / r) / t_ifov, 3)
                                for r in (20.0, 8.0, 2.0)},
    }
    return body, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mount")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    body, stats = build()
    stl = out / "saresq_sensor_mount.stl"
    body.export(stl)

    # PETG at ~1.27 g/cm3, 25% infill + 3 perimeters -> roughly 45% of solid.
    grams = stats["volume_cm3"] * 1.27 * 0.45
    stats["est_print_grams_petg"] = round(grams, 1)
    stats["stl"] = str(stl)

    for k, v in stats.items():
        print(f"  {k}: {v}")
    if not stats["watertight"]:
        raise SystemExit("mesh is NOT watertight -- a print shop will reject it")


if __name__ == "__main__":
    main()
