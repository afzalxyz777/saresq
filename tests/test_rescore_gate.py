"""The ground station's re-score acting as a gate on operator attention.

The rules under test are safety rules, not preferences: a second opinion may
promote a candidate but must never bury one, and nothing may be filtered away.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from saresq.store.db import Store                      # noqa: E402


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    yield s
    s.close()


def _target(s, tid, p):
    s.insert_target(target_id=tid, first_seen_ns=1, last_seen_ns=1,
                    p_final=p, n_passes=1,
                    **{"class": "HIGH" if p >= 0.6 else "MEDIUM"})
    return tid


def _rescore(s, tid, p, p_payload=None):
    mid = s.insert_media(target_id=tid, t_ns=1, kind="rgb_crop", sha256=f"s{tid}",
                         rel_path=f"{tid}.jpg", bytes=10, priority=p_payload or 0.0)
    s.insert_rescore(media_id=mid, target_id=tid, t_ns=1, p=p, n=1,
                     p_payload=p_payload, model="yolov8m.pt", imgsz=640)


def test_ground_score_can_promote_a_target(store):
    """The life-saving case: the aircraft's nano nearly let it go, the larger
    model on the ground says otherwise, and it moves to the top."""
    _target(store, 1, 0.80)
    _target(store, 2, 0.20)
    _rescore(store, 2, 0.95, p_payload=0.20)
    q = store.review_queue()
    assert [t["target_id"] for t in q] == [2, 1]
    assert q[0]["agreement"] == "GROUND_HIGHER"


def test_a_low_ground_score_never_buries_the_aircraft(store):
    """The safety rule. A disagreement is FLAGGED, but the target keeps its
    place, because being wrong here means walking past a casualty."""
    _target(store, 1, 0.90)
    _target(store, 2, 0.50)
    _rescore(store, 1, 0.01, p_payload=0.90)      # ground says false alarm
    q = store.review_queue()
    assert [t["target_id"] for t in q] == [1, 2]  # still first
    assert q[0]["agreement"] == "GROUND_LOWER"
    assert q[0]["gate_p"] == pytest.approx(0.90)


def test_nothing_is_ever_filtered_out(store):
    for i, p in enumerate([0.9, 0.5, 0.05, 0.001], start=1):
        _target(store, i, p)
    _rescore(store, 4, 0.0, p_payload=0.001)
    assert len(store.review_queue()) == 4


def test_agreement_labels(store):
    _target(store, 1, 0.70); _rescore(store, 1, 0.66, p_payload=0.70)
    _target(store, 2, 0.70)                                   # no rescore yet
    _target(store, 3, 0.10); _rescore(store, 3, 0.90, p_payload=0.10)
    _target(store, 4, 0.70); _rescore(store, 4, 0.02, p_payload=0.70)
    got = {t["target_id"]: t["agreement"] for t in store.review_queue()}
    assert got == {1: "AGREE", 2: "NO_RESCORE", 3: "GROUND_HIGHER", 4: "GROUND_LOWER"}


def test_best_rescore_is_used_not_the_last(store):
    """A target has many crops. The gate must see the strongest evidence, not
    whichever row happened to be written most recently."""
    _target(store, 1, 0.10)
    for p in (0.05, 0.88, 0.11):
        _rescore(store, 1, p, p_payload=0.10)
    q = store.review_queue()
    assert q[0]["rescore_p"] == pytest.approx(0.88)
    assert q[0]["gate_p"] == pytest.approx(0.88)


def test_judged_targets_leave_the_queue(store):
    _target(store, 1, 0.80)
    _rescore(store, 1, 0.90, p_payload=0.80)
    assert len(store.review_queue()) == 1
    store.insert_verdict(target_id=1, verdict="SURVIVOR", operator="test", t_ns=2)
    assert store.review_queue() == []
