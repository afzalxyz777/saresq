"""Task 0.10 acceptance test: reproduce the Section 10.8 worked example to 3 decimals."""
import pytest

from saresq.fuse.classes import classify
from saresq.fuse.ledger import Target


def test_worked_example_two_passes_different_bands():
    """Pass 1 (band 1, 20 m) MEDIUM thermal-only, pass 2 (band 0, 10 m) RGB-confirmed.

    Expected P_final = 0.960 (Section 10.8).
    """
    t = Target(lat=22.5, lon=88.3)
    t.add_pass(alt_band=1, p_k=0.45)  # 20 m survey pass: thermal only
    t.add_pass(alt_band=0, p_k=0.88)  # 10 m orbit pass: RGB confirms, new band -> w=1.0
    assert t.p_final == pytest.approx(0.960, abs=1e-3)
    assert t.decision() == "CONFIRM"
    cls = classify(p=t.p_final, has_rgb=True, iou=0.40, hits=4, z_peak=5.8)
    assert cls == "HIGH"


def test_worked_example_repeat_band_is_discounted():
    """Same scenario, but pass 2 re-observes the same band as pass 1 (w_repeat=0.7).

    Expected P_final = 0.897 (Section 10.8): still HIGH, but below the 0.90
    confirm threshold, so a third pass would be requested.
    """
    t = Target(lat=22.5, lon=88.3)
    t.add_pass(alt_band=1, p_k=0.45)
    t.add_pass(alt_band=1, p_k=0.88)  # same band as pass 1 -> w=0.7
    assert t.p_final == pytest.approx(0.897, abs=1e-3)
    assert t.decision() == "REOBSERVE_LOWER"


def test_low_confidence_target_is_rejected():
    t = Target(lat=0, lon=0)
    t.add_pass(alt_band=1, p_k=0.02)
    assert t.decision() == "REJECT"


def test_classify_downgrades_high_score_without_rgb_to_medium():
    assert classify(p=0.95, has_rgb=False, iou=0.0, hits=3, z_peak=5.0) == "MEDIUM"


def test_classify_flags_rgb_without_thermal_corroboration_as_low():
    assert classify(p=0.9, has_rgb=True, iou=0.5, hits=3, z_peak=1.0, z_t=2.5) == "LOW"
