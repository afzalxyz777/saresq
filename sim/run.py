"""Run the policy comparison and sensitivity sweep (Section 15.3, task 0.11).

    python -m sim.run

writes results/simulator_comparison.md and results/simulator_sensitivity.md.
"""
from __future__ import annotations

import math

import numpy as np

from sim.policies import run_trial
from sim.world import generate_world

POLICIES = ["lawnmower", "adaptive", "lawnmower_low"]
N_TRIALS = 100


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of a 95% CI over the finite values only.

    time_to_first/all_confirmation are +inf when the event never happened in
    a trial; including inf in mean/std would just produce nan, so those
    trials are excluded here and reported separately via _finite_fraction.
    """
    arr = np.array([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return math.inf, 0.0
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, 0.0
    sem = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, 1.96 * sem


def _finite_fraction(values: list[float]) -> float:
    return sum(1 for v in values if math.isfinite(v)) / len(values)


def run_policy_comparison(n_trials: int = N_TRIALS, seed: int = 0, sensing_overrides: dict | None = None):
    metrics = {p: {
        "survivors_confirmed": [], "distractors_confirmed": [], "survivors_missed": [],
        "area_covered_m2": [], "battery_remaining_pct": [],
        "time_to_first_confirmation_s": [], "time_to_confirm_all_s": [],
    } for p in POLICIES}

    for trial_i in range(n_trials):
        world = generate_world(np.random.default_rng(seed * 100_000 + trial_i))
        for policy in POLICIES:
            rng = np.random.default_rng(seed * 100_000 + trial_i + hash(policy) % 1000)
            result = run_trial(rng, world, policy, sensing_overrides=sensing_overrides)
            for key in metrics[policy]:
                metrics[policy][key].append(getattr(result, key))

    summary = {}
    for policy in POLICIES:
        summary[policy] = {key: _mean_ci95(vals) for key, vals in metrics[policy].items()}
        summary[policy]["pct_trials_first_confirmed"] = (
            _finite_fraction(metrics[policy]["time_to_first_confirmation_s"]) * 100, 0.0
        )
        summary[policy]["pct_trials_all_confirmed"] = (
            _finite_fraction(metrics[policy]["time_to_confirm_all_s"]) * 100, 0.0
        )
    return summary, metrics


def format_summary_table(summary: dict) -> str:
    headers = ["policy", "survivors_confirmed", "distractors_confirmed", "survivors_missed",
               "area_covered_m2", "battery_remaining_pct",
               "pct_trials_first_confirmed", "time_to_first_confirmation_s (when it happens)",
               "pct_trials_all_confirmed", "time_to_confirm_all_s (when it happens)"]
    key_map = {
        "time_to_first_confirmation_s (when it happens)": "time_to_first_confirmation_s",
        "time_to_confirm_all_s (when it happens)": "time_to_confirm_all_s",
    }
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for policy, stats in summary.items():
        row = [policy]
        for header in headers[1:]:
            key = key_map.get(header, header)
            mean, ci = stats[key]
            if header.startswith("pct_"):
                row.append(f"{mean:.0f}%")
            elif math.isinf(mean):
                row.append("never")
            else:
                row.append(f"{mean:.2f} ± {ci:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def sensitivity_sweep(seed: int = 1, n_trials: int = 50) -> str:
    """Sweep the parameters that flatter the adaptive policy across x0.5 and x2."""
    baseline, _ = run_policy_comparison(n_trials=n_trials, seed=seed)
    baseline_rank = sorted(POLICIES, key=lambda p: -baseline[p]["survivors_confirmed"][0])

    sweeps = {
        "g_heavy_scale (concealment recall)": [0.5, 2.0],
        "ambient_k_scale (hot-band derating)": [0.5, 2.0],
        "rgb_fp_scale (distractor RGB false-positive rate)": [0.5, 2.0],
    }
    lines = [f"Baseline ranking by survivors confirmed: {baseline_rank}\n"]
    any_reversal = False
    for param, factors in sweeps.items():
        key = param.split(" ")[0]
        for factor in factors:
            summary, _ = run_policy_comparison(n_trials=n_trials, seed=seed, sensing_overrides={key: factor})
            rank = sorted(POLICIES, key=lambda p: -summary[p]["survivors_confirmed"][0])
            reversed_ = rank != baseline_rank
            any_reversal = any_reversal or reversed_
            lines.append(f"{param} x{factor}: ranking={rank}{'  <-- REVERSED' if reversed_ else ''}")
    lines.append(f"\nAny ranking reversal under a 2x sweep: {any_reversal}")
    return "\n".join(lines)


if __name__ == "__main__":
    import pathlib
    import time

    t0 = time.time()
    summary, _ = run_policy_comparison()
    table = format_summary_table(summary)
    elapsed = time.time() - t0

    sweep_text = sensitivity_sweep()

    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path("results/simulator_comparison.md").write_text(
        f"# Simulator: policy comparison ({N_TRIALS} trials/policy)\n\n{table}\n\n"
        f"Wall time for {N_TRIALS} trials x {len(POLICIES)} policies: {elapsed:.2f}s\n"
    )
    pathlib.Path("results/simulator_sensitivity.md").write_text(
        f"# Simulator: sensitivity sweep (Section 15.3)\n\n```\n{sweep_text}\n```\n"
    )
    print(table)
    print(f"\n{elapsed:.2f}s for {N_TRIALS} trials x {len(POLICIES)} policies")
    print("\n" + sweep_text)
