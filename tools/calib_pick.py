"""Fit the thermal->RGB affine from frame sets saved by the live demo.

    # 1. on the phone/browser, put the target board in view and press
    #    "Save frame set" two or three times, moving the board between presses
    # 2. pull the files and fit:
    python tools/calib_pick.py results/calib/live_thermal_*.npy
    # 3. check the overlay it writes, then deploy:
    bash tools/deploy_demo.sh

WHY THE THERMAL SIDE IS NOT CLICKED
One thermal pixel is ~45 RGB pixels (register.NOMINAL_SCALE_RGB_PER_THERMAL_PX).
A human click that is one pixel off in a 32x24 image is therefore a 45 px error
in the fit -- larger than the RANSAC threshold the fit is judged against, so
hand-clicking the thermal side cannot produce a calibration that passes its own
acceptance check. Instead each target is found automatically and located by an
intensity-weighted centroid, which is sub-pixel: a 4-pixel blob whose energy sits
slightly right of centre returns col=12.31, not 12. Only the RGB side is clicked,
where one pixel of click error is one pixel of error.

WHY ROBUST z AND NOT mean/std
A calibration board is, by construction, several strong outliers in a small
array. Mean and standard deviation are computed FROM those outliers, so the
targets inflate the very threshold meant to find them -- with 8 hot mugs in a
32x24 frame the std is dominated by the mugs and they stop being outliers.
Median and 1.4826*MAD are not moved by them, which is the same reasoning
saresq/gate/contrast.py uses and the opposite of the demo's own gate_stats().

WHY COLD TARGETS WORK BETTER THAN HOT ONES
Ice in a tray sits ~20 K below room temperature and nothing else in an indoor
scene does. A mug of hot water competes with radiators, laptops, lights and
people, and cools visibly during the capture. Pass --sign cold for ice, hot for
mugs, or both to take whichever is present.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from saresq.calib.register import fit_affine, apply_affine  # noqa: E402

TH_SCALE = 22          # thermal display zoom; 32x24 -> 704x528
RGB_MAX_W = 1100       # RGB display width cap, so the window fits a laptop screen


def robust_z(t: np.ndarray) -> np.ndarray:
    med = float(np.median(t))
    mad = float(np.median(np.abs(t - med)))
    return (t - med) / max(1.4826 * mad, 0.15)   # 0.15 K floor: the MLX's own noise


def find_targets(t: np.ndarray, sign: str, z_t: float, min_px: int, max_n: int):
    """Sub-pixel centroids of the calibration targets, in thermal pixel coords."""
    z = robust_z(t)
    masks = []
    if sign in ("hot", "both"):
        masks.append(z >= z_t)
    if sign in ("cold", "both"):
        masks.append(z <= -z_t)

    found = []
    for m in masks:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) < min_px:
                continue
            ys, xs = np.nonzero(labels == i)
            w = np.abs(z[ys, xs])            # weight by contrast, not temperature:
            w = w / w.sum()                  # a 0 C reference and a 60 C one weigh alike
            found.append({
                "col": float((xs * w).sum()),
                "row": float((ys * w).sum()),
                "px": int(stats[i, cv2.CC_STAT_AREA]),
                "z": float(z[ys, xs][np.abs(z[ys, xs]).argmax()]),
                "T": float(t[ys, xs].mean()),
            })
    # Largest first: a big, well-resolved target has a more trustworthy centroid
    # than a 2-pixel speck, and the user clicks the confident ones first.
    found.sort(key=lambda b: -b["px"])
    return found[:max_n]


def draw_thermal(t: np.ndarray, targets) -> np.ndarray:
    lo, hi = float(np.percentile(t, 1)), float(np.percentile(t, 99))
    if hi - lo < 6.0:
        hi = lo + 6.0
    norm = np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap(
        cv2.resize((norm * 255).astype(np.uint8), (32 * TH_SCALE, 24 * TH_SCALE),
                   interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)
    for k, b in enumerate(targets):
        x, y = int((b["col"] + .5) * TH_SCALE), int((b["row"] + .5) * TH_SCALE)
        cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.circle(img, (x, y), 15, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(k + 1), (x + 17, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, str(k + 1), (x + 17, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def collect_pair(npy: pathlib.Path, rgb_path: pathlib.Path, args):
    """Show one frame set; return the (thermal_pt, rgb_pt) pairs the user clicked."""
    t = np.load(npy)
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        print(f"  !! cannot read {rgb_path.name}, skipping")
        return []

    targets = find_targets(t, args.sign, args.z, args.min_px, args.max_targets)
    if not targets:
        print(f"  !! no targets found in {npy.name} at z>={args.z} "
              f"({args.sign}); is the board in the thermal view?")
        return []

    print(f"\n  {npy.name}: {len(targets)} target(s) "
          f"[{', '.join(f'{b[chr(84)]:.1f}C' for b in targets)}]")
    print("  click each NUMBERED target in the RGB window, in order 1..N")
    print("  keys:  u = undo last   n = done with this frame   s = skip frame   q = quit")

    scale = min(1.0, RGB_MAX_W / rgb.shape[1])
    disp0 = cv2.resize(rgb, None, fx=scale, fy=scale) if scale < 1 else rgb.copy()
    clicks: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(targets):
            clicks.append((x / scale, y / scale))     # store FULL-RES coordinates

    cv2.namedWindow("thermal (reference)", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("RGB - click here", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("RGB - click here", on_mouse)
    cv2.imshow("thermal (reference)", draw_thermal(t, targets))

    while True:
        d = disp0.copy()
        for k, (fx, fy) in enumerate(clicks):
            x, y = int(fx * scale), int(fy * scale)
            cv2.drawMarker(d, (x, y), (80, 235, 130), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(d, str(k + 1), (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(d, str(k + 1), (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (80, 235, 130), 2, cv2.LINE_AA)
        nxt = len(clicks) + 1
        msg = (f"click target {nxt} of {len(targets)}" if nxt <= len(targets)
               else "all clicked - press n for the next frame")
        cv2.rectangle(d, (0, 0), (d.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(d, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("RGB - click here", d)

        k = cv2.waitKey(30) & 0xFF
        if k == ord('u') and clicks:
            clicks.pop()
        elif k == ord('n'):
            break
        elif k == ord('s'):
            clicks = []
            break
        elif k == ord('q'):
            cv2.destroyAllWindows()
            return [("QUIT", None)]

    # Only complete correspondences are usable; a half-clicked frame contributes
    # its finished pairs and nothing else.
    return [((targets[i]["col"], targets[i]["row"]), clicks[i])
            for i in range(min(len(clicks), len(targets)))]


def write_overlay(npy: pathlib.Path, rgb_path: pathlib.Path, matrix, out: pathlib.Path):
    """Render the fused view the calibration makes possible -- and the proof it worked."""
    t, rgb = np.load(npy), cv2.imread(str(rgb_path))
    if rgb is None:
        return None
    H, W = rgb.shape[:2]
    # Map every thermal pixel corner into RGB space, then warp. The affine is
    # thermal_px -> rgb_px, which is exactly what warpAffine wants.
    z = robust_z(t)
    heat = np.clip((z + 2.0) / 8.0, 0, 1)
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    warped = cv2.warpAffine(heat, np.asarray(matrix, dtype=np.float32), (W, H),
                            flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    # Alpha rises with thermal contrast, so a uniform wall stays transparent and
    # only real heat paints over the RGB detail.
    a = cv2.warpAffine(np.clip(np.abs(z) / 6.0, 0, 1).astype(np.float32),
                       np.asarray(matrix, dtype=np.float32), (W, H))[..., None]
    blend = (rgb * (1 - a * 0.65) + warped * (a * 0.65)).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), blend)
    return out


def _spread_select(frames, args, k: int):
    """Greedy farthest-point subset of the captures, by where the target landed.

    Thirty-eight captures of a pan waved around a room are mostly redundant: an
    affine has six parameters, and ten points clustered in the middle constrain
    it no better than one. What conditions the fit is the SPREAD of the points,
    so this keeps the frames that are farthest apart in thermal coordinates and
    drops the near-duplicates. The user clicks a third as many frames for
    essentially the same information -- and the clicks they do make are the ones
    that carry it, out at the edges where extrapolation error would be worst.
    """
    import numpy as _np

    located = []
    for f in frames:
        try:
            tg = find_targets(_np.load(f), args.sign, args.z, args.min_px, 1)
        except Exception:
            continue
        if tg:
            located.append((f, tg[0]["col"], tg[0]["row"]))
    if len(located) <= k:
        return [f for f, _, _ in located]

    pts = _np.array([[c, r] for _, c, r in located], dtype=float)
    # Seed with the point farthest from the centroid: start at an extreme rather
    # than in the middle, or the first few picks just crawl outward from centre.
    chosen = [int(_np.argmax(_np.linalg.norm(pts - pts.mean(0), axis=1)))]
    while len(chosen) < k:
        d = _np.min(_np.linalg.norm(pts[:, None] - pts[chosen][None], axis=2), axis=1)
        d[chosen] = -1.0
        chosen.append(int(_np.argmax(d)))

    picked = [located[i] for i in sorted(chosen, key=lambda i: located[i][0].name)]
    print(f"selected {len(picked)} of {len(located)} frames for spread "
          f"(col {min(p[1] for p in picked):.1f}-{max(p[1] for p in picked):.1f}, "
          f"row {min(p[2] for p in picked):.1f}-{max(p[2] for p in picked):.1f})")
    return [f for f, _, _ in picked]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("thermal", nargs="+", help="live_thermal_*.npy from the demo")
    ap.add_argument("--sign", default="both", choices=["hot", "cold", "both"])
    ap.add_argument("--z", type=float, default=3.0, help="robust-z threshold for a target")
    ap.add_argument("--min-px", type=int, default=2, help="reject specks below this area")
    ap.add_argument("--max-targets", type=int, default=12, help="per frame")
    ap.add_argument("--ransac-px", type=float, default=3.0)
    ap.add_argument("--out", default=str(REPO / "saresq" / "calib" / "thermal_to_rgb.json"))
    ap.add_argument("--serial", default="", help="stamped into the JSON")
    ap.add_argument("--select", type=int, default=0,
                    help="click only this many frames, chosen for maximum spread")
    args = ap.parse_args()

    th_pts: list = []
    rgb_pts: list = []
    used: list[tuple[pathlib.Path, pathlib.Path]] = []

    frames = [pathlib.Path(p) for p in args.thermal]
    if args.select and len(frames) > args.select:
        frames = _spread_select(frames, args, args.select)

    for npy in frames:
        rgb_path = npy.with_name(npy.name.replace("thermal", "rgb")).with_suffix(".jpg")
        if not rgb_path.exists():
            print(f"  !! no RGB partner for {npy.name} (looked for {rgb_path.name})")
            continue
        pairs = collect_pair(npy, rgb_path, args)
        if pairs and pairs[0][0] == "QUIT":
            break
        if pairs:
            used.append((npy, rgb_path))
        for tp, rp in pairs:
            th_pts.append(tp)
            rgb_pts.append(rp)
        print(f"  -> {len(pairs)} correspondence(s); {len(th_pts)} total")

    cv2.destroyAllWindows()

    if len(th_pts) < 3:
        sys.exit(f"\nneed at least 3 correspondences to fit an affine, got {len(th_pts)}")
    if len(th_pts) < 12:
        print(f"\nNOTE: {len(th_pts)} points. The procedure recommends 12+ spread "
              "across the field of view including corners; fewer will fit but the "
              "residual understates the true error away from the points.")

    reg = fit_affine(np.array(th_pts), np.array(rgb_pts), ransac_thresh_px=args.ransac_px)
    n_in = int(reg.inlier_mask.sum())
    print(f"\n{'='*58}\nfit: {n_in}/{reg.n_points} inliers")
    print(f"residual: {reg.residual_rgb_px:.2f} RGB px "
          f"= {reg.residual_thermal_px:.3f} thermal px")
    print("matrix (thermal_px -> rgb_px):")
    for row in reg.matrix:
        print("   " + "  ".join(f"{v: 10.4f}" for v in row))

    # Acceptance: sub-thermal-pixel. Above 1.0 the reprojected box lands on the
    # wrong thermal cell entirely and every iou/d_c feature built on it is noise.
    if reg.residual_thermal_px > 1.0:
        print("\n** REJECT: residual exceeds one thermal pixel. Re-capture with the "
              "targets better spread, and check nothing moved between frames. **")
    elif reg.residual_thermal_px > 0.5:
        print("\n** MARGINAL: usable, but add a frame with targets near the corners. **")
    else:
        print("\nACCEPTED (sub-half-thermal-pixel).")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reg.save(out, plate_serial=args.serial, date=time.strftime("%Y-%m-%d"))
    print(f"\nwrote {out}")

    for npy, rgb_path in used[:3]:
        dest = REPO / "results" / "calib" / f"overlay_{npy.stem}.jpg"
        if write_overlay(npy, rgb_path, reg.matrix, dest):
            print(f"wrote {dest}")
    print("\nOpen the overlay(s). If the heat sits ON the warm objects, it is right.")


if __name__ == "__main__":
    main()
