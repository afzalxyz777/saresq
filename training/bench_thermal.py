#!/usr/bin/env python3
"""Reproduce the thermal-stage numbers the paper quotes.

    .venv/bin/python training/bench_thermal.py

Writes results/thermal/thermal_bench.json. Everything here is deterministic
(fixed seed, synthetic ground truth) so the paper's figures can be regenerated
on any machine without the payload attached -- which matters, because the
alternative is a table of numbers nobody can check.

Three measurements:

  drizzle   A known Gaussian source is sampled by a 32x24 grid at sub-pixel
            offsets. Because the truth is synthetic we can ask the only
            question that matters: does the reconstruction land the CENTROID
            closer to where the source actually is than plain interpolation
            does, and is it quieter?

  motion    False-alarm rate on a still scene, and the smallest limb shift the
            detector still catches. A motion flag that fires on noise is worse
            than none: it teaches an operator to ignore it.

  subpage   The MLX90640 interleaves two chessboard subpages. On an intact
            frame both sample the same scene, so their means agree; a read
            straddling a refresh returns halves from different moments and the
            means separate. Measured here on the payload captures in
            results/thermal/.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from saresq.thermal.drizzle import Drizzle, estimate_shift
from saresq.thermal.motion import ThermalMotion

ROWS, COLS = 24, 32
RNG = np.random.default_rng(20260918)
NETD_K = 0.10                     # MLX90640 datasheet noise-equivalent delta-T

#: The two interleaved chessboard subpages. Pixel (r,c) belongs to subpage
#: (r + c) % 2 -- that is the sensor's own readout pattern, not a choice.
_RR, _CC = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
SUB_A = ((_RR + _CC) % 2 == 0)
SUB_B = ~SUB_A


AMP, BASE = 6.0, 22.0
HALF_MAX = BASE + AMP / 2.0       # the fixed level every centroid is cut at


def scene(dy: float, dx: float, amp: float = AMP, sigma: float = 1.6,
          base: float = BASE) -> np.ndarray:
    """A Gaussian heat source, sampled at a sub-pixel offset, plus NETD noise."""
    r = np.arange(ROWS)[:, None] - (ROWS / 2 + dy)
    c = np.arange(COLS)[None, :] - (COLS / 2 + dx)
    f = base + amp * np.exp(-(r ** 2 + c ** 2) / (2 * sigma ** 2))
    return f + RNG.normal(0.0, NETD_K, f.shape)


def centroid(field: np.ndarray, half_max: float) -> tuple[float, float]:
    """Intensity-weighted centroid of the hottest region, in source pixels.

    The threshold is a FIXED TEMPERATURE, not a percentile of the field. A
    percentile cuts at a different physical level on a 32x24 grid than on a
    64x48 one -- it selects a fixed FRACTION of pixels, and the two grids do
    not have the same number to begin with. Scoring three reconstructions at
    different resolutions against a percentile therefore measures the metric,
    not the reconstruction: it initially ranked the raw 32x24 frame as more
    accurate than either upsampling of it, which is not a result, it is a bug.
    """
    f = np.nan_to_num(field, nan=float(np.nanmin(field)))
    w = np.clip(f - half_max, 0.0, None)
    if w.sum() <= 0:
        return (np.nan, np.nan)
    rr, cc = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    sy, sx = f.shape[0] / ROWS, f.shape[1] / COLS
    # Pixel CENTRES, not indices. A fine-grid pixel i covers source coordinate
    # (i + 0.5)/s - 0.5; using i/s instead leaves a constant 0.25 px offset at
    # s = 2, which is larger than every difference this benchmark is trying to
    # resolve. It made both upsampled reconstructions look eight times worse
    # than the raw frame they were built from -- the tell that the metric, and
    # not the method, was what was being measured.
    return (float((w * (rr + 0.5)).sum() / w.sum()) / sy - 0.5,
            float((w * (cc + 0.5)).sum() / w.sum()) / sx - 0.5)


def bench_drizzle(n_trials: int = 40, n_frames: int = 8, scale: int = 2) -> dict:
    """Drizzle against single-frame interpolation, on synthetic ground truth."""
    import cv2
    e_single, e_driz, n_single, n_driz, trusted = [], [], [], [], 0
    e_raw, e_oracle, n_oracle = [], [], []
    for _ in range(n_trials):
        ty, tx = RNG.uniform(-2, 2), RNG.uniform(-2, 2)   # true source offset
        frames = []
        for _k in range(n_frames):
            # Sub-pixel dither: without it drizzle has nothing extra to work
            # with and cannot beat interpolation even in principle.
            jy, jx = RNG.uniform(-0.5, 0.5), RNG.uniform(-0.5, 0.5)
            frames.append((scene(ty + jy, tx + jx), jy, jx))

        # --- single frame, NO upsampling: the raw 32x24 grid ---
        # Included because it is the comparison that flatters drizzle, and the
        # difference between the two baselines is the whole point. Against the
        # raw grid any finer sampling wins on centroid almost by construction;
        # against a bicubic upsample to the SAME grid, which is what the
        # payload would otherwise display, it is a fair fight.
        raw = frames[0][0]
        cy, cx = centroid(raw, HALF_MAX)
        e_raw.append(np.hypot(cy - (ROWS / 2 + ty + frames[0][1]),
                              cx - (COLS / 2 + tx + frames[0][2])))

        # --- single frame, bicubic up to the same grid ---
        one = frames[0][0]
        up = cv2.resize(one, (COLS * scale, ROWS * scale), interpolation=cv2.INTER_CUBIC)
        cy, cx = centroid(up, HALF_MAX)
        e_single.append(np.hypot(cy - (ROWS / 2 + ty + frames[0][1]),
                                 cx - (COLS / 2 + tx + frames[0][2])))
        n_single.append(float(np.std(up[:4, :4])))        # corner: source-free

        # --- drizzle, shifts estimated the way the payload estimates them ---
        d = Drizzle(shape=(ROWS, COLS), scale=scale)
        ref = frames[0][0]
        for f, _jy, _jx in frames:
            # estimate_shift returns (dx, dy) and add() takes (dx, dy). Passed
            # straight through, with no unpacking in between to get backwards.
            d.add(f, estimate_shift(ref, f))
        res = d.result()
        trusted += int(res.trustworthy)
        cy, cx = centroid(res.field, HALF_MAX)
        e_driz.append(np.hypot(cy - (ROWS / 2 + ty + frames[0][1]),
                               cx - (COLS / 2 + tx + frames[0][2])))
        n_driz.append(float(np.nanstd(res.field[:4, :4])))

        # --- drizzle with the TRUE dither, which the payload never has ---
        # This separates the method from its input. If the oracle run is good
        # and the estimated run is not, the limit is the shift estimator and
        # not drizzle; reporting only the estimated number would blame the
        # wrong component and point future work in the wrong direction.
        d2 = Drizzle(shape=(ROWS, COLS), scale=scale)
        j0y, j0x = frames[0][1], frames[0][2]
        for f, jy, jx in frames:
            # (reference - this frame), not the other way round: a feature at
            # position p in this frame sits at p + (j0 - j) in the reference.
            # Backwards, it doubles the dither instead of removing it -- the
            # same sign trap drizzle.py itself was caught by, and the reason
            # this oracle arm exists at all.
            d2.add(f, (j0x - jx, j0y - jy))     # (dx, dy) -- x FIRST
        r2 = d2.result()
        cy, cx = centroid(r2.field, HALF_MAX)
        e_oracle.append(np.hypot(cy - (ROWS / 2 + ty + j0y),
                                 cx - (COLS / 2 + tx + j0x)))
        n_oracle.append(float(np.nanstd(r2.field[:4, :4])))

    return {
        "n_trials": n_trials, "n_frames": n_frames, "scale": scale,
        "centroid_err_px_raw": round(float(np.mean(e_raw)), 4),
        "centroid_err_px_single": round(float(np.mean(e_single)), 4),
        "centroid_err_px_drizzle": round(float(np.mean(e_driz)), 4),
        "noise_k_single": round(float(np.mean(n_single)), 4),
        "noise_k_drizzle": round(float(np.mean(n_driz)), 4),
        "centroid_err_px_drizzle_true_shifts": round(float(np.mean(e_oracle)), 4),
        "noise_k_drizzle_true_shifts": round(float(np.mean(n_oracle)), 4),
        "trustworthy_frac": round(trusted / n_trials, 3),
    }


def bench_motion(n_still: int = 600) -> dict:
    """False alarms on a still scene, and the smallest shift still detected.

    Reseeded on entry. Without this the counts depend on how many random draws
    the drizzle benchmark happened to consume first, and the same script prints
    187 or 167 for the same measurement depending on the order its functions
    run in -- which is not a number anyone can check.
    """
    global RNG
    RNG = np.random.default_rng(20260918)
    # With and without ego-motion compensation. The gap between these two is
    # the whole finding: before the sub-pixel deadband, alignment was the
    # ONLY source of false movement on a still scene.
    fa = {}
    for label, comp in (("compensated", True), ("uncompensated", False)):
        m = ThermalMotion(compensate=comp)
        fa[label] = sum(m.add(scene(0.0, 0.0)).moved for _ in range(n_still))
    false_alarms = fa["compensated"]

    # Smallest detected limb shift. A warm limb-sized patch is displaced by a
    # growing number of pixels until the detector fires.
    detected_at = None
    for shift_px in (1, 2, 3, 4, 5, 6, 8):
        m2 = ThermalMotion()
        base = scene(0.0, 0.0)
        for _ in range(6):
            m2.add(base + RNG.normal(0, NETD_K, base.shape))
        limb = base.copy()
        limb[10:14, 12:12 + shift_px] += 1.6          # a limb moves into frame
        got = any(m2.add(limb + RNG.normal(0, NETD_K, base.shape)).moved
                  for _ in range(3))
        if got and detected_at is None:
            detected_at = shift_px
    # Deadband sweep. The paper quotes these, so they are measured here rather
    # than extrapolated from a percentage -- which is how a "173 / 600" that
    # nobody ever counted got as far as a draft.
    import saresq.thermal.motion as _M
    sweep, keep = {}, _M._ALIGN_FLOOR_PX
    try:
        for floor in (0.0, 0.5, 1.0):
            _M._ALIGN_FLOOR_PX = floor
            m = _M.ThermalMotion()
            sweep[f"{floor:.1f}"] = sum(m.add(scene(0.0, 0.0)).moved
                                        for _ in range(n_still))
    finally:
        _M._ALIGN_FLOOR_PX = keep

    return {
        "still_frames": n_still,
        "deadband_sweep_false_alarms": sweep,
        "false_alarms": false_alarms,
        "false_alarm_rate": round(false_alarms / n_still, 4),
        "false_alarms_uncompensated": fa["uncompensated"],
        "min_detected_shift_px": detected_at,
    }


def bench_subpage() -> dict:
    """Subpage mean agreement on the payload's own saved captures."""
    out = []
    for p in sorted(pathlib.Path("results/thermal").glob("*.npy")):
        f = np.load(p)
        if f.shape != (ROWS, COLS):
            f = f.reshape(ROWS, COLS)
        gap = abs(float(f[SUB_A].mean()) - float(f[SUB_B].mean()))
        out.append({"capture": p.name, "subpage_gap_k": round(gap, 4)})
    return {"captures": out, "threshold_k": 0.60}


def main() -> None:
    res = {
        "drizzle": bench_drizzle(),
        "motion": bench_motion(),
        "subpage": bench_subpage(),
        "netd_k": NETD_K,
        "seed": 20260918,
    }
    out = pathlib.Path("results/thermal/thermal_bench.json")
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
