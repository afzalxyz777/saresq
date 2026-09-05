"""Task 0.11 smoke test: the simulator runs fast and its central hypothesis holds
under the current (labelled-as-provisional) sensing-model calibration."""
import time

from sim.run import run_policy_comparison


def test_runs_100_trials_per_policy_in_under_a_minute():
    t0 = time.time()
    summary, _ = run_policy_comparison(n_trials=100)
    elapsed = time.time() - t0
    assert elapsed < 60
    assert set(summary.keys()) == {"lawnmower", "adaptive", "lawnmower_low"}


def test_adaptive_confirms_more_survivors_than_fixed_lawnmower():
    """Section 15.3's stated hypothesis: adaptive trades coverage for confirmations."""
    summary, _ = run_policy_comparison(n_trials=200, seed=42)
    adaptive_confirmed = summary["adaptive"]["survivors_confirmed"][0]
    lawnmower_confirmed = summary["lawnmower"]["survivors_confirmed"][0]
    adaptive_area = summary["adaptive"]["area_covered_m2"][0]
    lawnmower_area = summary["lawnmower"]["area_covered_m2"][0]

    assert adaptive_confirmed > lawnmower_confirmed
    assert adaptive_area <= lawnmower_area
