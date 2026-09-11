import json

import numpy as np
import pytest

from saresq.fuse.features import FEATURE_NAMES
from saresq.store.db import Store
from saresq.store.media import MediaStore

from training.export_verdicts import export


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    yield s
    s.close()


def _target(store, **kw):
    kw.setdefault("first_seen_ns", 1)
    kw.setdefault("last_seen_ns", 2)
    kw.setdefault("lat", 22.573)
    kw.setdefault("lon", 88.364)
    kw.setdefault("p_final", 0.96)
    kw.setdefault("class_", "HIGH")
    kw.setdefault("n_passes", 2)
    kw.setdefault("decision", "CONFIRM")
    return store.insert_target(**kw)


def test_a_verdict_never_mutates_the_machines_belief(store):
    tid = _target(store)
    before = store.get_target(tid)

    store.insert_verdict(target_id=tid, verdict="NOT_SURVIVOR", operator="alpha", t_ns=99,
                         p_final_at_verdict=before["p_final"],
                         decision_at_verdict=before["decision"], note="corrugated sheet")

    after = store.get_target(tid)
    assert after["p_final"] == before["p_final"] == 0.96
    assert after["decision"] == before["decision"] == "CONFIRM"
    # The disagreement is preserved on both sides, which is the point.
    v = store.latest_verdict(tid)
    assert v["verdict"] == "NOT_SURVIVOR" and v["decision_at_verdict"] == "CONFIRM"


def test_review_queue_is_confidence_ordered_and_drops_judged_targets(store):
    lo = _target(store, p_final=0.22, class_="LOW", decision="REOBSERVE_LOWER")
    hi = _target(store, p_final=0.96)
    mid = _target(store, p_final=0.55, class_="MEDIUM")

    assert [t["target_id"] for t in store.review_queue()] == [hi, mid, lo]

    store.insert_verdict(target_id=hi, verdict="SURVIVOR", operator="a", t_ns=1)
    assert [t["target_id"] for t in store.review_queue()] == [mid, lo]


def test_latest_features_returns_the_most_recent_pass(store):
    tid = _target(store)
    p1 = store.insert_pass(target_id=tid, alt_m=20.0, p_pass=0.45)
    p2 = store.insert_pass(target_id=tid, alt_m=10.0, p_pass=0.88)
    store.insert_pass_features(p1, tid, 100, [0.1] * len(FEATURE_NAMES))
    store.insert_pass_features(p2, tid, 200, [0.9] * len(FEATURE_NAMES))

    feats = store.latest_features_for_target(tid)
    assert feats == [0.9] * len(FEATURE_NAMES)


def test_export_builds_a_labelled_set_and_drops_unusable_rows(store, tmp_path):
    good_pos = _target(store, p_final=0.96, decision="CONFIRM")
    good_neg = _target(store, p_final=0.91, decision="CONFIRM")
    unsure = _target(store, p_final=0.5, decision="REOBSERVE_LOWER")
    featureless = _target(store, p_final=0.8, decision="CONFIRM")

    for tid, vals in ((good_pos, 0.9), (good_neg, 0.4), (unsure, 0.5)):
        pid = store.insert_pass(target_id=tid, alt_m=20.0)
        store.insert_pass_features(pid, tid, 10, [vals] * len(FEATURE_NAMES))

    def judge(tid, verdict):
        t = store.get_target(tid)
        store.insert_verdict(
            target_id=tid, verdict=verdict, operator="alpha", t_ns=1,
            p_final_at_verdict=t["p_final"], decision_at_verdict=t["decision"],
            features_json=json.dumps(store.latest_features_for_target(tid))
            if store.latest_features_for_target(tid) else None,
        )

    judge(good_pos, "SURVIVOR")
    judge(good_neg, "NOT_SURVIVOR")   # machine said CONFIRM -> a real disagreement
    judge(unsure, "UNSURE")
    judge(featureless, "SURVIVOR")

    out = tmp_path / "labels"
    s = export(str(store.conn.execute("PRAGMA database_list").fetchone()["file"]), str(out))

    assert s["n"] == 2 and s["positive"] == 1 and s["negative"] == 1
    assert s["dropped_unsure"] == 1
    assert s["dropped_no_features"] == 1
    assert s["disagreements"] == 1

    d = np.load(out / "fusion_labels.npz", allow_pickle=False)
    assert d["X"].shape == (2, len(FEATURE_NAMES))
    assert sorted(d["y"].tolist()) == [0, 1]
    assert list(d["feature_names"]) == FEATURE_NAMES

    csv_text = (out / "fusion_labels.csv").read_text()
    assert "p_rgb" in csv_text and "NOT_SURVIVOR" in csv_text


def test_dispatched_counts_as_a_positive_label(store, tmp_path):
    tid = _target(store)
    pid = store.insert_pass(target_id=tid, alt_m=12.0)
    store.insert_pass_features(pid, tid, 5, [0.3] * len(FEATURE_NAMES))
    store.insert_verdict(target_id=tid, verdict="DISPATCHED", operator="a", t_ns=1,
                         p_final_at_verdict=0.96, decision_at_verdict="CONFIRM",
                         features_json=json.dumps([0.3] * len(FEATURE_NAMES)))

    s = export(str(store.conn.execute("PRAGMA database_list").fetchone()["file"]), str(tmp_path / "l"))
    assert s["n"] == 1 and s["positive"] == 1 and s["disagreements"] == 0
