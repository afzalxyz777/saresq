"""Task 0.6 acceptance tests (Master Spec v3.0, Section 19.1)."""
import numpy as np
import pytest

from saresq.gate.blobs import extract_blobs
from saresq.gate.budget import crop_budget
from saresq.gate.contrast import compute_contrast, gate_mask

FRAME_SHAPE = (24, 32)  # (rows=v, cols=u), matches MLX90640


def _flat_frame(temp_c: float, shape=FRAME_SHAPE) -> np.ndarray:
    return np.full(shape, temp_c, dtype=float)


def test_one_pixel_warm_spot_on_cool_background_is_gated():
    """A 1-pixel +3 K spot on a 30-degree background is found at z above 2.5."""
    frame = _flat_frame(30.0)
    frame[10, 15] = 33.0
    t_bg, sigma, z, dT = compute_contrast(frame)
    assert abs(z[10, 15]) > 2.5
    mask = gate_mask(z, dT)
    blobs = extract_blobs(z, dT, mask)
    assert len(blobs) == 1
    assert blobs[0].sign == +1
    assert blobs[0].area_t == 1
    assert blobs[0].z_peak > 2.5


def test_cool_spot_on_hot_background_is_gated_with_negative_sign():
    """A 2x3 -4 K spot on a 45-degree background (regime 3) is found with sign -1."""
    frame = _flat_frame(45.0)
    frame[5:7, 10:13] = 41.0  # 2 rows x 3 cols, -4 K
    t_bg, sigma, z, dT = compute_contrast(frame)
    mask = gate_mask(z, dT)
    blobs = extract_blobs(z, dT, mask)
    assert len(blobs) == 1
    blob = blobs[0]
    assert blob.sign == -1
    assert blob.area_t == 6
    assert abs(blob.z_peak) > 2.5


def test_one_sided_gate_misses_cool_spot_on_hot_background():
    """The one-sided ('warmer than') gate is blind to regime 3; two-sided is not."""
    frame = _flat_frame(45.0)
    frame[5:7, 10:13] = 41.0
    t_bg, sigma, z, dT = compute_contrast(frame)
    one_sided_mask = gate_mask(z, dT, two_sided=False)
    assert not one_sided_mask.any()
    two_sided_mask = gate_mask(z, dT, two_sided=True)
    assert two_sided_mask.any()


def test_flat_noisy_frame_produces_no_blobs():
    """A flat frame with 0.1 K RMS noise produces no blobs (dt_min_k rejects it)."""
    rng = np.random.default_rng(seed=0)
    frame = _flat_frame(30.0) + rng.normal(0.0, 0.1, size=FRAME_SHAPE)
    t_bg, sigma, z, dT = compute_contrast(frame)
    mask = gate_mask(z, dT)
    blobs = extract_blobs(z, dT, mask)
    assert len(blobs) == 0


@pytest.mark.parametrize("z_max,expected", [(4.0, 2), (1.5, 6), (2.75, 4)])
def test_crop_budget_widens_as_contrast_weakens(z_max, expected):
    """The crop budget goes from 2 to 6 as z_max falls from 4 to 1.5."""
    assert crop_budget(z_max) == expected
