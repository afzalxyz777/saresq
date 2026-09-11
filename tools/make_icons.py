"""Generate the installed-app icons.

Run once; the PNGs are committed. Kept as a script rather than done by hand so
the mark can be regenerated at any size without redrawing it, and so the icon
is provably the same shape as the diamond in the console's top bar.

    python tools/make_icons.py
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent / "saresq" / "dashboard" / "static" / "icons"
GROUND = (11, 8, 4)      # BGR of --ground #04080B
CYAN = (236, 205, 63)    # BGR of --friend #3FCDEC
RING = (52, 42, 26)      # BGR of --rule #1A2A34


def draw(size: int, safe: float) -> np.ndarray:
    """A diamond inside a ring. `safe` is the fraction of the canvas the mark
    is allowed to occupy -- maskable icons get cropped to a circle by Android,
    so they need a wider margin than a plain one."""
    ss = 4  # supersample, then downscale: cheap antialiasing without a font
    n = size * ss
    img = np.full((n, n, 3), GROUND, np.uint8)
    c = n // 2
    r_ring = int(c * safe * 0.94)
    cv2.circle(img, (c, c), r_ring, RING, max(1, int(n * 0.012)), cv2.LINE_AA)
    r = int(c * safe * 0.60)
    pts = np.array([[c, c - r], [c + r, c], [c, c + r], [c - r, c]], np.int32)
    cv2.fillPoly(img, [pts], CYAN, cv2.LINE_AA)
    # Notch the diamond into a chevron so the mark reads at 48 px on a phone.
    r2 = int(r * 0.42)
    pts2 = np.array([[c, c - r2], [c + r2, c], [c, c + r2], [c - r2, c]], np.int32)
    cv2.fillPoly(img, [pts2], GROUND, cv2.LINE_AA)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for size in (180, 192, 512):
        p = OUT / f"icon-{size}.png"
        cv2.imwrite(str(p), draw(size, 0.92))
        made.append(p)
    p = OUT / "icon-maskable-512.png"
    cv2.imwrite(str(p), draw(512, 0.66))
    made.append(p)
    for p in made:
        print(f"{p.relative_to(OUT.parent.parent.parent.parent)}  {p.stat().st_size} B")


if __name__ == "__main__":
    main()
