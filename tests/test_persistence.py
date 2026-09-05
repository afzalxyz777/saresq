"""Blob persistence across frames (Section 5.3's `persist` feature)."""
from saresq.gate.blobs import Blob
from saresq.gate.persistence import BlobPersistenceTracker
from saresq.track.egomotion import predict_transform


def _blob_at(u: int, v: int) -> Blob:
    return Blob(area_t=1, z_peak=3.0, z_mean=3.0, dT_peak=3.0, sign=1, ecc=0.0, bbox_t=(u, v, u, v))


def test_stationary_blob_persists_across_frames_with_no_ego_motion():
    tracker = BlobPersistenceTracker()
    counts_f1 = tracker.update([_blob_at(10, 10)], transform=None)
    counts_f2 = tracker.update([_blob_at(10, 10)], transform=None)
    counts_f3 = tracker.update([_blob_at(10, 10)], transform=None)
    assert counts_f1 == [1]
    assert counts_f2 == [2]
    assert counts_f3 == [3]


def test_blob_that_disappears_and_reappears_resets_persistence():
    tracker = BlobPersistenceTracker()
    tracker.update([_blob_at(10, 10)], transform=None)
    tracker.update([_blob_at(10, 10)], transform=None)
    tracker.update([], transform=None)  # vanished for one frame
    counts = tracker.update([_blob_at(10, 10)], transform=None)
    assert counts == [1]  # persistence resets, not carried across the gap


def test_ego_motion_compensated_ground_stationary_blob_still_persists():
    """A ground-stationary person drifts across the thermal frame as the
    aircraft moves; without compensation this would look like a new blob
    every frame (mirrors the A4 RGB-tracker ablation, at thermal scale)."""
    tracker_compensated = BlobPersistenceTracker(radius_thermal_px=1.5)
    tracker_uncompensated = BlobPersistenceTracker(radius_thermal_px=1.5)

    transform = predict_transform(
        v_mps=5.0, course_rad=0.0, yaw_rad=0.0,
        roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
        dt_s=0.25, altitude_m=20.0, focal_px=1359.3,
    )
    # This RGB-space shift, divided by the along-track scale (36.8), is the
    # thermal-pixel drift per frame the blob exhibits on the sensor.
    from saresq.gate.persistence import SCALE_ALONG
    thermal_drift_per_frame = transform.ty / SCALE_ALONG
    assert thermal_drift_per_frame > 0.5  # a real, non-trivial drift to compensate for

    comp_counts, uncomp_counts = [], []
    for k in range(5):
        v = 10 + round(k * thermal_drift_per_frame)
        comp_counts.append(tracker_compensated.update([_blob_at(10, v)], transform)[0])
        uncomp_counts.append(tracker_uncompensated.update([_blob_at(10, v)], None)[0])

    assert comp_counts == [1, 2, 3, 4, 5]
    assert uncomp_counts[-1] == 1  # each frame looks like a brand-new blob
