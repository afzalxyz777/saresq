"""Task 0.9 acceptance test (A4 ablation): ego-motion compensation vs fragmentation.

A stationary target is tracked at a simulated 14 m/s, 20 m altitude — the
worst case from Section 3.4, where the RGB image shifts 95 px per 100 ms.
"""
import math

from saresq.track.egomotion import predict_transform
from saresq.track.sort import EgoMotionTracker

F_PX = 1359.3
ALT_M = 20.0
SPEED_MPS = 14.0
DT_S = 0.1
N_FRAMES = 50
CENTER = (820.0, 616.0)
BOX_SIZE = (34.0, 34.0)


def _run(compensate: bool):
    tracker = EgoMotionTracker(compensate=compensate)
    forward_mag = F_PX * SPEED_MPS * DT_S / ALT_M  # ~95 px, matches Section 3.4's table
    u0, v0 = 800.0, 600.0

    transform = predict_transform(
        v_mps=SPEED_MPS, course_rad=0.0, yaw_rad=0.0,
        roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
        dt_s=DT_S, altitude_m=ALT_M, focal_px=F_PX,
    )

    for k in range(N_FRAMES):
        # True (noisy) detection of a stationary ground target as the aircraft moves.
        v_true = v0 + k * forward_mag
        det = (u0, v_true, *BOX_SIZE)
        tracker.step([det], transform, CENTER, dt=DT_S)

    return tracker


def test_compensated_tracker_holds_one_track_with_no_fragmentation():
    tracker = _run(compensate=True)
    assert tracker.n_tracks_created == 1
    assert len(tracker.tracks) == 1
    track = tracker.tracks[0]
    assert track.confirmed
    assert track.hits == N_FRAMES
    assert track.misses == 0


def test_uncompensated_tracker_fragments():
    tracker = _run(compensate=False)
    # A 95 px/frame apparent drift with a 34 px box gives ~zero IoU after
    # frame one: the tracker can never re-associate and spawns a fresh
    # tentative track almost every frame.
    assert tracker.n_tracks_created > 10
    assert not any(t.confirmed for t in tracker.tracks)


def test_forward_motion_matches_spec_table_value():
    """Section 3.4: at 14 m/s and 20 m, the RGB pixel shift per 100 ms is ~95 px."""
    forward_mag = F_PX * SPEED_MPS * DT_S / ALT_M
    assert math.isclose(forward_mag, 95.15, rel_tol=0.01)
