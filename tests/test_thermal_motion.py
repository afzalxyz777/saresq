"""Thermal motion detection: movement is the evidence that a heat source is alive.

The false-alarm tests matter more than the detection test. A motion flag that
fires on a still room teaches an operator to ignore it, and an ignored flag is
worse than no flag.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from saresq.thermal.motion import ThermalMotion   # noqa: E402

H, W, NETD = 24, 32, 0.10


def _room(leg_y=None):
    t = np.full((H, W), 31.0)
    t[8:16, 10:20] = 32.2            # a body under a thin blanket
    if leg_y is not None:
        t[leg_y:leg_y + 3, 18:24] = 34.0   # an exposed limb
    return t


def _noisy(t, rng):
    return t + rng.normal(0, NETD, t.shape)


def test_a_still_room_never_reports_motion():
    rng = np.random.default_rng(0)
    m = ThermalMotion()
    fired = sum(m.add(_noisy(_room(12), rng), t_s=i * 0.25).moved for i in range(40))
    assert fired == 0


def test_a_moved_limb_is_detected():
    rng = np.random.default_rng(0)
    m = ThermalMotion()
    hits = []
    for i in range(30):
        r = m.add(_noisy(_room(12 if i < 15 else 16), rng), t_s=i * 0.25)
        if i >= 15:
            hits.append(r)
    moved = [h for h in hits if h.moved]
    assert moved, "a limb shifting by four pixels must register"
    assert moved[0].peak_dK > 1.0
    assert moved[0].area_px >= 4


def test_motion_is_transient_by_design():
    """A discrete move is reported while it is happening, not forever. Holding
    the flag is the caller's job, so that 'moved 3 s ago' stays distinguishable
    from 'moving now'."""
    rng = np.random.default_rng(0)
    m = ThermalMotion()
    for i in range(15):
        m.add(_noisy(_room(12), rng), t_s=i * 0.25)
    for i in range(15, 40):
        r = m.add(_noisy(_room(16), rng), t_s=i * 0.25)
    assert not r.moved            # long settled by frame 40


def test_slow_drift_is_not_motion():
    """A room or a sensor warming up shifts every pixel together. That is
    drift, and reporting it as a living person would be a lie."""
    rng = np.random.default_rng(1)
    m = ThermalMotion()
    fired = sum(m.add(_noisy(_room(12) + i * 0.02, rng), t_s=i * 0.25).moved
                for i in range(40))
    assert fired <= 1


def test_payload_panning_is_mostly_compensated():
    """Whole-scene movement is the aircraft, not the casualty. Phase
    correlation on a near-featureless thermal frame is imperfect, so this
    asserts a large reduction rather than perfection -- in flight the GNSS/IMU
    transform supplies the shift directly and does better."""
    rng = np.random.default_rng(2)
    base = _room(12)

    def run(compensate):
        m = ThermalMotion(compensate=compensate)
        fired = 0
        for i in range(30):
            M = np.float32([[1, 0, -i * 0.4], [0, 1, 0]])
            t = cv2.warpAffine(base.astype(np.float32), M, (W, H),
                               borderMode=cv2.BORDER_REPLICATE)
            r = m.add(_noisy(t.astype(float), rng), t_s=i * 0.25)
            if i >= 8:
                fired += r.moved
        return fired

    assert run(True) < run(False)


def test_no_result_until_the_lag_window_is_full():
    m = ThermalMotion(lag_frames=4)
    rng = np.random.default_rng(0)
    for i in range(4):
        assert not m.add(_noisy(_room(12), rng)).moved


def test_alignment_does_not_manufacture_motion_on_a_still_scene():
    """The sub-pixel deadband in _estimate. This one shipped broken.

    On a low-texture thermal frame -- a warm body on a flat background, i.e.
    the normal case -- phase correlation scatters by up to ~1.1 px between two
    frames of a scene that has NOT moved. Warping the older frame by that
    imaginary offset shears every gradient in it, and the difference image then
    carries real signal at the edge of the warm blob. Measured at 23 false
    "movements" in 80 still frames before the deadband; ~0.5% after.

    This matters more than a tidy number: a false motion flag promotes
    BODY_HEAT to LIVE_BODY, which is the console asserting that a warm object
    is a LIVING person. A radiator does not need a rescue team.
    """
    rng = np.random.default_rng(4242)
    rr = np.arange(24)[:, None] - 12.0
    cc = np.arange(32)[None, :] - 16.0
    base = 22.0 + 6.0 * np.exp(-(rr ** 2 + cc ** 2) / (2 * 1.6 ** 2))

    m = ThermalMotion()
    false_alarms = sum(m.add(base + rng.normal(0, 0.10, base.shape)).moved
                       for _ in range(200))
    assert false_alarms <= 4, f"{false_alarms}/200 false alarms on a still scene"


def test_a_real_shift_is_still_compensated():
    """The deadband must not disable ego-motion compensation.

    Motion worth correcting is large -- 95 px per 100 ms at 14 m/s and 20 m --
    so rejecting sub-pixel estimates costs nothing. Guard that the large case
    still works, or the fix above would have traded one failure for a worse one.
    """
    rng = np.random.default_rng(5)
    # A wider world, from which the sensor sees a 32-px window. Panning the
    # WINDOW is a true translation; np.roll would instead wrap one edge around
    # and hand the detector genuinely new content to be right about.
    world = rng.normal(22.0, 2.0, (24, 48))
    view = lambda x0: world[:, x0:x0 + 32]           # noqa: E731

    m = ThermalMotion()
    for _ in range(6):
        m.add(view(4) + rng.normal(0, 0.10, (24, 32)))
    # The whole scene slides 3 px: that is the aircraft, not a casualty.
    assert not m.add(view(7) + rng.normal(0, 0.10, (24, 32))).moved
