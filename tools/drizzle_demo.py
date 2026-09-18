"""Run drizzle over recorded thermal frames and show what it bought.

    .venv/bin/python tools/drizzle_demo.py results/calib/live_thermal_*.npy

Writes a side-by-side panel (raw / Lanczos / drizzled) plus the quality
numbers, so the claim can be checked against real captures rather than against
a synthetic scene. The 38 frames from the calibration session are hand-panned
over a warm target, which is exactly the sub-pixel dither drizzle needs.

Frames are grouped into consecutive windows: the static-scene assumption only
holds over a second or two, so combining a whole session in one stack would
smear rather than sharpen.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from saresq.thermal.drizzle import Drizzle, estimate_shift   # noqa: E402

PALETTE = cv2.COLORMAP_INFERNO


def colourise(field: np.ndarray, size: tuple[int, int], lo: float, hi: float,
              interp: int = cv2.INTER_NEAREST) -> np.ndarray:
    norm = np.clip((field - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap(
        cv2.resize((norm * 255).astype(np.uint8), size, interpolation=interp), PALETTE)


def label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]),
                      (0, 0, 0), -1)
        cv2.putText(img, sub, (9, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--window", type=int, default=8, help="frames per stack")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--pixfrac", type=float, default=0.65)
    ap.add_argument("--out", default="results/thermal/drizzle_demo.jpg")
    args = ap.parse_args()

    paths = sorted(pathlib.Path(p) for p in args.frames)
    frames = [np.load(p).astype(np.float64) for p in paths]
    print(f"{len(frames)} frames of {frames[0].shape}")

    best = None
    for start in range(0, len(frames) - args.window + 1, max(args.window // 2, 1)):
        chunk = frames[start:start + args.window]
        d = Drizzle(chunk[0].shape, scale=args.scale, pixfrac=args.pixfrac,
                    max_frames=len(chunk))
        shifts = []
        for f in chunk:
            s = estimate_shift(chunk[0], f)
            shifts.append(s)
            d.add(f, s)
        r = d.result()
        span = float(np.ptp([s[0] for s in shifts])), float(np.ptp([s[1] for s in shifts]))
        print(f"  frames {start:3}-{start+len(chunk)-1:3}  diversity {r.diversity:.3f}  "
              f"coverage {r.coverage:.3f}  shift span {span[0]:+.2f},{span[1]:+.2f} px  "
              f"{'USABLE' if r.trustworthy else 'rejected'}")
        if r.trustworthy and (best is None or r.diversity > best[1].diversity):
            best = (chunk, r)

    if best is None:
        print("\nNo window had enough sub-pixel diversity to claim super-resolution.")
        print("That is the correct answer for a payload that did not move between")
        print("frames -- pan it slowly by a fraction of a pixel and run again.")
        return

    chunk, r = best
    raw = chunk[0]
    # One common temperature scale across all three panels, or the comparison
    # is between palettes rather than between reconstructions.
    lo, hi = float(np.percentile(raw, 1)), float(np.percentile(raw, 99))
    if hi - lo < 4.0:
        hi = lo + 4.0
    size = (r.field.shape[1] * 12, r.field.shape[0] * 12)

    panels = [
        label(colourise(raw, size, lo, hi, cv2.INTER_NEAREST),
              "RAW  32x24", "one frame, nearest-neighbour"),
        label(colourise(raw, size, lo, hi, cv2.INTER_LANCZOS4),
              "LANCZOS", "one frame, interpolated - adds no information"),
        label(colourise(r.field, size, lo, hi, cv2.INTER_NEAREST),
              f"DRIZZLE x{r.scale}",
              f"{r.n_frames} frames, diversity {r.diversity:.2f}, pixfrac {r.pixfrac}"),
    ]
    out = np.hstack(panels)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)

    noise_raw = float(np.std(raw - cv2.medianBlur(raw.astype(np.float32), 3)))
    f32 = r.field.astype(np.float32)
    noise_dz = float(np.std(f32 - cv2.medianBlur(f32, 3)))
    print(f"\nhigh-frequency residual: raw {noise_raw:.3f} K -> drizzled {noise_dz:.3f} K")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
