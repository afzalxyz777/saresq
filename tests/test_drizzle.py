"""Drizzle regression tests.

The synthetic scene here is the one the module's measured claims come from, so
these tests are what stops those numbers quietly rotting.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from saresq.thermal.drizzle import Drizzle, estimate_shift

S, H, W = 8, 24, 32
NETD = 0.10
DITHER = [(0, 0), (.5, 0), (0, .5), (.5, .5),
          (.25, .75), (.75, .25), (.25, .25), (.75, .75)]


def _scene(cx_px: float, cy_px: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H * S, 0:W * S]
    t = np.full((H * S, W * S), 30.0)
    t[((yy - cy_px * S) ** 2 + (xx - cx_px * S) ** 2) < (1.3 * S) ** 2] = 38.0
    return t


def _sample(truth, sx, sy, rng=None):
    M = np.float32([[1, 0, -sx * S], [0, 1, -sy * S]])
    w = cv2.warpAffine(truth, M, (W * S, H * S), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_REFLECT)
    f = w.reshape(H, S, W, S).mean(axis=(1, 3))
    return f if rng is None else f + rng.normal(0, NETD, f.shape)


def _centroid(field, scale):
    z = field - np.median(field)
    z[z < 0.4 * z.max()] = 0
    yy, xx = np.mgrid[0:field.shape[0], 0:field.shape[1]]
    tot = z.sum()
    return (xx * z).sum() / tot / scale, (yy * z).sum() / tot / scale


def test_radiometry_is_preserved():
    """Output is a weighted MEAN, so temperatures keep their units. A drizzle
    that returned a sum would be correct for photons and wrong for kelvin."""
    truth = _scene(16, 12)
    d = Drizzle((H, W), scale=2, max_frames=64)
    for sx, sy in DITHER:
        d.add(_sample(truth, sx, sy), (sx, sy))
    f = d.result().field
    assert 29.9 <= float(np.nanmin(f)) <= 30.1
    assert float(np.nanmax(f)) <= 38.05           # never exceeds the true peak


def test_uniform_field_stays_uniform():
    d = Drizzle((H, W), scale=2, max_frames=64)
    for sx, sy in DITHER:
        d.add(np.full((H, W), 27.5), (sx, sy))
    f = d.result().field
    assert np.allclose(f, 27.5, atol=1e-6)


def test_subpixel_dither_halves_the_centroid_error():
    """The operational claim: 8 dithered frames locate a warm target more than
    twice as precisely as one raw frame."""
    rng = np.random.default_rng(3)
    e1, e8 = [], []
    for _ in range(12):
        cx, cy = rng.uniform(8, 24), rng.uniform(6, 18)
        truth = _scene(cx, cy)
        e1.append(np.hypot(*np.subtract(_centroid(_sample(truth, 0, 0, rng), 1), (cx, cy))))
        d = Drizzle((H, W), scale=2, max_frames=64)
        for sx, sy in DITHER:
            d.add(_sample(truth, sx, sy, rng), (sx, sy))
        e8.append(np.hypot(*np.subtract(_centroid(d.result().field, 2), (cx, cy))))
    assert np.mean(e1) / np.mean(e8) > 1.8


def test_integer_shifts_are_reported_as_untrustworthy():
    """A payload that never moved sub-pixel has no super-resolution to claim,
    and the result must say so rather than looking plausible."""
    truth = _scene(16, 12)
    d = Drizzle((H, W), scale=2, max_frames=64)
    for k in range(8):
        d.add(_sample(truth, k, 0), (k, 0))
    r = d.result()
    assert r.diversity < 0.05
    assert not r.trustworthy


def test_good_dither_is_trustworthy():
    truth = _scene(16, 12)
    d = Drizzle((H, W), scale=2, max_frames=64)
    for sx, sy in DITHER:
        d.add(_sample(truth, sx, sy), (sx, sy))
    r = d.result()
    assert r.diversity > 0.9 and r.coverage > 0.99 and r.trustworthy


def test_estimate_shift_recovers_a_known_offset():
    """Tolerance is the MEASURED p90 (0.184 px) rounded up, not an aspiration.
    Phase correlation on a 32x24 frame has a floor near an eighth of a pixel."""
    truth = _scene(16, 12)
    ref = _sample(truth, 0, 0)
    for sx, sy in ((0.5, 0.0), (0.0, 0.75), (1.25, -0.5)):
        dx, dy = estimate_shift(ref, _sample(truth, sx, sy))
        assert np.hypot(dx - sx, dy - sy) < 0.3


def test_estimated_shifts_are_good_enough_to_keep_the_gain():
    """End to end with NOTHING handed in: shifts recovered from the frames
    themselves. This is the bench path, where there is no GPS fix to derive
    motion from, and it must not give away the improvement."""
    rng = np.random.default_rng(11)
    e1, e8 = [], []
    for _ in range(10):
        cx, cy = rng.uniform(8, 24), rng.uniform(6, 18)
        truth = _scene(cx, cy)
        frames = [_sample(truth, sx, sy, rng) for sx, sy in DITHER]
        e1.append(np.hypot(*np.subtract(_centroid(frames[0], 1), (cx, cy))))
        d = Drizzle((H, W), scale=2, max_frames=64)
        for f in frames:
            d.add(f, estimate_shift(frames[0], f))
        e8.append(np.hypot(*np.subtract(_centroid(d.result().field, 2), (cx, cy))))
    assert np.mean(e1) / np.mean(e8) > 1.8


def test_shift_sign_matches_the_documented_convention():
    """Guards the bug that cost real time: an inverted sign does not crash, it
    doubles the dither and silently produces a blurrier image than plain
    interpolation."""
    truth = _scene(16, 12)
    ref2 = truth.reshape(H * 2, S // 2, W * 2, S // 2).mean(axis=(1, 3))

    def build(sign):
        d = Drizzle((H, W), scale=2, max_frames=64)
        for sx, sy in DITHER:
            d.add(_sample(truth, sx, sy), (sign * sx, sign * sy))
        return float(np.sqrt(np.nanmean((d.result().field - ref2) ** 2)))

    assert build(+1) < build(-1)


def test_rejects_bad_geometry():
    with pytest.raises(ValueError):
        Drizzle((H, W), pixfrac=0.0)
    with pytest.raises(ValueError):
        Drizzle((H, W), scale=0)
    with pytest.raises(ValueError):
        Drizzle((H, W)).add(np.zeros((5, 5)))
