"""Section 11.1: degrading public thermal data to look MLX90640-like."""
import numpy as np
import pytest

from training.thermal_degradation import degrade_box, degrade_thermal, scale_for_target_person_size


def test_scale_shrinks_a_typical_person_box_to_the_target_size():
    scale = scale_for_target_person_size(box_w_px=8.0, box_h_px=11.0, target_px=2.0)
    avg = 8.0 * scale, 11.0 * scale
    assert 1.0 <= sum(avg) / 2 <= 3.0


def test_scale_never_upsamples():
    """A person already smaller than target_px must not be enlarged."""
    scale = scale_for_target_person_size(box_w_px=1.0, box_h_px=1.0, target_px=2.0)
    assert scale == 1.0


def test_degrade_thermal_shrinks_and_adds_noise():
    rng = np.random.default_rng(0)
    clean = np.zeros((100, 100), dtype=np.float32)
    clean[40:60, 40:60] = 200.0  # a bright square on a dark background

    degraded = degrade_thermal(clean, scale=0.3, rng=rng)
    assert degraded.shape == (30, 30)

    degraded_twice = degrade_thermal(clean, scale=0.3, rng=np.random.default_rng(1))
    assert not np.allclose(degraded, degraded_twice)  # noise actually varies run to run


def test_degrade_box_scales_consistently_with_the_image():
    box = (100.0, 100.0, 108.0, 111.0)
    scaled = degrade_box(box, scale=0.25)
    assert scaled == pytest.approx((25.0, 25.0, 27.0, 27.75))
