#!/usr/bin/env python3
"""Regenerate docs/paper/fig/parallax.png (Fig. 5) from equation (9).

    .venv/bin/python tools/make_parallax_fig.py

The figure existed before this script did, drawn at a 27 mm baseline that
appears in no CAD file -- tools/make_payload_cad.py has always said 28.0. The
plot is now derived from the same constant the part is cut from, so the two
cannot drift again.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASELINE_M = 0.028          # tools/make_payload_cad.py: BASELINE
FOCAL_PX = 1359.3           # configs/pipeline.yaml: sensors.rgb.focal_px
THERMAL_PX_IN_RGB = 45.0    # 45:1 resolution ratio, Section IV-D
BUDGET_PX = 0.5             # configs/pipeline.yaml: registration.max_residual
FLY_M = 20.0                # mission.survey_alt_m


def residual_thermal_px(z_cal_m, z_fly_m=FLY_M):
    """Equation (9), converted from visible pixels to thermal pixels."""
    d_rgb = FOCAL_PX * BASELINE_M * np.abs(1.0 / z_cal_m - 1.0 / z_fly_m)
    return d_rgb / THERMAL_PX_IN_RGB


def main() -> None:
    z = np.linspace(0.5, 8.0, 600)
    r = residual_thermal_px(z)

    fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=300)
    ax.plot(z, r, color="#111", lw=1.4)
    ax.axhline(BUDGET_PX, color="#b03a2e", lw=1.0, ls="--")
    ax.text(7.9, BUDGET_PX + 0.02, "0.5 px budget", ha="right", va="bottom",
            fontsize=6.5, color="#b03a2e")

    for zc, mark in ((1.0, "o"), (2.0, "s")):
        rc = residual_thermal_px(zc)
        ax.plot([zc], [rc], mark, ms=4, color="#111", zorder=5)
        ax.annotate(f"{zc:.0f} m: {rc:.3f} px",
                    xy=(zc, rc), xytext=(zc + 0.45, rc + 0.10),
                    fontsize=6.5,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#555"))

    ax.set_xlabel("calibration standoff $Z_c$  [m]", fontsize=7.5)
    ax.set_ylabel("residual  [thermal px]", fontsize=7.5)
    ax.set_xlim(0.5, 8.0)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, lw=0.4, alpha=0.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.2)

    out = pathlib.Path("docs/paper/fig/parallax.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    for zc in (1.0, 2.0, 3.0, 4.0):
        print(f"  Zc = {zc:.0f} m -> {residual_thermal_px(zc):.4f} thermal px")


if __name__ == "__main__":
    main()
