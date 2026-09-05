"""Figures for the deck/video: policy comparison (Section 15.3, task 0.11).

    python -m sim.plots
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from sim.run import POLICIES, run_policy_comparison

POLICY_LABELS = {
    "lawnmower": "Fixed lawnmower\n(20 m, 5 m/s)",
    "adaptive": "Adaptive\n(perception-driven)",
    "lawnmower_low": "Fixed lawnmower\n(12 m everywhere)",
}
COLORS = {"lawnmower": "#8899aa", "adaptive": "#1f8a4c", "lawnmower_low": "#c2762a"}


def make_comparison_figure(summary: dict, out_path: str = "results/simulator_comparison.png") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    labels = [POLICY_LABELS[p] for p in POLICIES]
    colors = [COLORS[p] for p in POLICIES]

    # Panel 1: survivors confirmed vs missed vs distractors confirmed.
    ax = axes[0]
    confirmed = [summary[p]["survivors_confirmed"][0] for p in POLICIES]
    confirmed_ci = [summary[p]["survivors_confirmed"][1] for p in POLICIES]
    missed = [summary[p]["survivors_missed"][0] for p in POLICIES]
    fp = [summary[p]["distractors_confirmed"][0] for p in POLICIES]
    x = np.arange(len(POLICIES))
    width = 0.27
    ax.bar(x - width, confirmed, width, yerr=confirmed_ci, label="survivors confirmed", color=colors, capsize=3)
    ax.bar(x, missed, width, label="survivors never a candidate", color=colors, alpha=0.4, hatch="//")
    ax.bar(x + width, fp, width, label="distractors confirmed (false alarm)", color=colors, alpha=0.7, hatch="xx")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Outcomes per mission (mean of 100 trials)")
    ax.legend(fontsize=7, loc="upper right")

    # Panel 2: time to first confirmation.
    ax = axes[1]
    times = [summary[p]["time_to_first_confirmation_s"][0] for p in POLICIES]
    times = [0 if not np.isfinite(t) else t for t in times]
    pct = [summary[p]["pct_trials_first_confirmed"][0] for p in POLICIES]
    bars = ax.bar(x, times, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("seconds")
    ax.set_title("Time to first confirmation\n(when a confirmation happens)")
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, f"{p:.0f}% of trials",
                 ha="center", fontsize=7)

    # Panel 3: coverage vs battery remaining (the core trade-off).
    ax = axes[2]
    area = [summary[p]["area_covered_m2"][0] / 1000 for p in POLICIES]  # thousand m^2
    battery = [summary[p]["battery_remaining_pct"][0] for p in POLICIES]
    ax2 = ax.twinx()
    ax.bar(x - width / 2, area, width, color=colors, alpha=0.9)
    ax2.bar(x + width / 2, battery, width, color=colors, alpha=0.4, hatch="..")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("area covered (1000 m²)")
    ax2.set_ylabel("battery remaining (%)")
    ax.set_title("Coverage vs. battery remaining\n(solid = coverage, dotted = battery)")

    fig.suptitle("Perception-driven adaptive search vs. fixed lawnmower survey (Section 15.3 simulator)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pathlib.Path(out_path).parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    summary, _ = run_policy_comparison()
    make_comparison_figure(summary)
