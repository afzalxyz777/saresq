import numpy as np
import pytest

from saresq.dashboard import app as dash
from saresq.store.db import Store
from saresq.store.media import MediaStore


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    store = Store(str(db))
    media = MediaStore(tmp_path / "media", store)

    tid = store.insert_target(first_seen_ns=1, last_seen_ns=2, lat=22.573, lon=88.364,
                              pos_err_m=2.5, p_final=0.96, class_="HIGH", n_passes=1,
                              decision="CONFIRM")
    pid = store.insert_pass(target_id=tid, alt_m=10.0, p_pass=0.88)
    store.insert_pass_features(pid, tid, 10, [0.5] * 18)

    crop = np.full((32, 32, 3), 120, dtype=np.uint8)
    jpg_id = media.put_rgb_crop(crop, target_id=tid, pass_id=pid, priority=0.96)
    patch_id = media.put_thermal_patch(np.full((24, 32), 305.0), target_id=tid, pass_id=pid, priority=0.96)
    store.close()

    dash.app.config.update(DB_PATH=str(db), TESTING=True)
    # Deliberately RELATIVE, with the CWD moved: this is the exact shape of the
    # bug where Flask resolved the path against the package dir instead.
    monkeypatch.chdir(tmp_path)
    dash.app.config["MEDIA_DIR"] = "media"

    return dash.app.test_client(), tid, jpg_id, patch_id


def test_relative_media_dir_resolves_against_cwd_not_the_package(rig):
    client, _, jpg_id, _ = rig
    r = client.get(f"/media/{jpg_id}")
    assert r.status_code == 200, "a relative --media-dir must still serve blobs"
    assert r.mimetype == "image/jpeg"
    assert r.data[:2] == b"\xff\xd8", "should be real JPEG bytes"


def test_thermal_patch_renders_to_png(rig):
    client, _, _, patch_id = rig
    r = client.get(f"/media/{patch_id}/render")
    assert r.status_code == 200 and r.mimetype == "image/png"
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_refuses_a_non_thermal_media_id(rig):
    client, _, jpg_id, _ = rig
    assert client.get(f"/media/{jpg_id}/render").status_code == 404


def test_review_queue_carries_evidence_and_drops_judged_targets(rig):
    client, tid, _, _ = rig
    q = client.get("/api/review/queue").get_json()
    assert len(q) == 1 and q[0]["target_id"] == tid
    assert {m["kind"] for m in q[0]["media"]} == {"rgb_crop", "thermal_patch"}
    assert len(q[0]["passes"]) == 1

    ok = client.post("/api/verdict", json={"target_id": tid, "verdict": "SURVIVOR", "operator": "alpha"})
    assert ok.status_code == 200
    assert client.get("/api/review/queue").get_json() == []


def test_verdict_freezes_features_and_preserves_the_machine_decision(rig):
    client, tid, _, _ = rig
    client.post("/api/verdict", json={"target_id": tid, "verdict": "NOT_SURVIVOR",
                                      "operator": "alpha", "note": "corrugated sheet"})
    v = client.get("/api/verdicts").get_json()[0]
    assert v["decision_at_verdict"] == "CONFIRM" and v["p_final_at_verdict"] == 0.96
    assert v["note"] == "corrugated sheet"
    assert len(__import__("json").loads(v["features_json"])) == 18

    # The machine's own row is untouched.
    t = [x for x in client.get("/api/targets").get_json() if x["target_id"] == tid][0]
    assert t["p_final"] == 0.96 and t["decision"] == "CONFIRM"


def test_stats_count_a_machine_operator_disagreement(rig):
    client, tid, _, _ = rig
    client.post("/api/verdict", json={"target_id": tid, "verdict": "NOT_SURVIVOR", "operator": "a"})
    s = client.get("/api/review/stats").get_json()
    assert s["judged"] == 1 and s["disagree"] == 1 and s["agree"] == 0
    assert s["agreement"] == 0.0


@pytest.mark.parametrize("payload", [
    {"target_id": "one", "verdict": "SURVIVOR"},
    {"target_id": 1, "verdict": "MAYBE"},
    {"verdict": "SURVIVOR"},
])
def test_verdict_rejects_malformed_input(rig, payload):
    client, _, _, _ = rig
    assert client.post("/api/verdict", json=payload).status_code == 400


def test_verdict_on_an_unknown_target_is_404(rig):
    client, _, _, _ = rig
    assert client.post("/api/verdict", json={"target_id": 9999, "verdict": "SURVIVOR"}).status_code == 404


# ---------------------------------------------------------------------------
# radar
# ---------------------------------------------------------------------------
@pytest.fixture()
def radar_client(rig, monkeypatch):
    """A fresh radar service per test.

    The service is a module-level singleton on purpose -- a tracker that is
    rebuilt every poll can never coast -- so tests have to clear it or they
    inherit whichever aircraft the previous test left flying.
    """
    monkeypatch.setattr(dash, "_radar", None, raising=False)
    return rig[0]


def test_radar_snapshot_has_a_usable_picture(radar_client):
    r = radar_client.get("/api/radar")
    assert r.status_code == 200
    d = r.get_json()
    for key in ("platform", "contacts", "link", "search", "events", "aoi", "gcs"):
        assert key in d, f"/api/radar must carry {key}"
    assert d["simulated"] is True, "a simulated picture must always say so"
    assert d["link"]["state"] in ("UP", "DEGRADED", "DOWN")
    assert len(d["search"]["grid"]) == d["search"]["nx"] * d["search"]["ny"]


def test_radar_uses_the_stores_targets_when_it_has_them(radar_client):
    d = radar_client.get("/api/radar").get_json()
    assert d["source"] == "store", "the radar must rediscover real targets, not invent some"


def test_radar_fault_injection_round_trips(radar_client):
    r = radar_client.post("/api/radar/fault", json={"fault": "link", "seconds": 30})
    assert r.status_code == 200
    assert radar_client.get("/api/radar").get_json()["faults"]["link"] is not None
    radar_client.post("/api/radar/fault", json={"fault": "link"})
    assert radar_client.get("/api/radar").get_json()["faults"]["link"] is None


def test_radar_fault_rejects_an_unknown_name(radar_client):
    assert radar_client.post("/api/radar/fault", json={"fault": "nope"}).status_code == 400


def test_radar_page_and_pwa_assets_are_served(radar_client):
    assert radar_client.get("/radar").status_code == 200
    sw = radar_client.get("/sw.js")
    assert sw.status_code == 200
    # Scope: a worker under /static/ could never control the pages themselves.
    assert sw.headers.get("Service-Worker-Allowed") == "/"
    assert "no-cache" in sw.headers.get("Cache-Control", "")
    mf = radar_client.get("/static/manifest.webmanifest")
    assert mf.status_code == 200
    assert mf.get_json()["start_url"] == "/radar"
