"""The payload -> ground-station bridge (saresq/dashboard/payload.py).

Runs against a stub HTTP server shaped like tools/live_pipeline.py, so the
whole join is covered with no Pi, no camera and no network.
"""
from __future__ import annotations

import http.server
import json
import socketserver
import threading
import time

import pytest

from saresq.dashboard.payload import MERGE_RADIUS_M, PayloadLink, _haversine_m, _klass
from saresq.store.db import Store

STATS = {
    "thermal": {"min": 30.1, "max": 36.4, "spread": 6.3, "fps": 4.0},
    "gate": {"z_max": 4.2, "z_t": 2.5, "fired": True, "n_blobs": 2},
    "detect": {"n": 1, "ms": 310, "model": "yolov8n_coco_640_w8a32.tflite"},
    "gps": {"quality": 1, "sats": 7, "lat": 22.5726, "lon": 88.3639,
            "hdop": 1.2, "in_view": 9, "state": "3D fix"},
    "hazard": {"ok": True, "top": "flooded_areas", "p": 0.91},
    "temp": "58.2'C", "up": "12:00:00",
}
EVENTS = [
    {"id": 1, "s": "ab", "conf": 0.81, "z": 4.2, "n": 1, "tmax": 34.1,
     "lat": 22.5726, "lon": 88.3639, "sats": 7},
    # ~3 m from #1: the same person seen twice, not a second find.
    {"id": 2, "s": "ab", "conf": 0.44, "z": 3.1, "n": 1, "tmax": 33.2,
     "lat": 22.57262, "lon": 88.36392, "sats": 7},
    {"id": 3, "s": "ab", "conf": 0.67, "z": 3.9, "n": 1, "tmax": 35.0,
     "lat": 22.5800, "lon": 88.3700, "sats": 6},
    {"id": 4, "s": "ab", "conf": 0.55, "z": 3.3, "n": 1, "tmax": 33.8,
     "lat": None, "lon": None, "sats": 0},
]


#: Mutable so a test can simulate the payload being power cycled: the handler
#: reads it on every request, and rewriting it is exactly what a restarted
#: aircraft looks like from the ground.
LIVE = {"events": EVENTS}


@pytest.fixture()
def payload_server():
    LIVE["events"] = EVENTS
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep pytest output clean
            pass

        def do_GET(self):  # noqa: N802
            body = json.dumps(LIVE["events"] if self.path == "/events" else STATS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _run(host, db_path, origin=None, wait=2.5):
    link = PayloadLink(host, store_factory=lambda: Store(str(db_path)), events_s=0.5)
    if origin:
        link.set_origin(*origin)
    link.start()
    deadline = time.time() + wait
    while time.time() < deadline and link.live().get("ingested", 0) < len(EVENTS):
        time.sleep(0.05)
    link.stop()
    return link


def test_live_state_is_shaped_for_the_ui(payload_server, tmp_path):
    link = _run(payload_server, tmp_path / "t.db")
    s = link.live()
    # "configured" is added by the Flask route, not by live(); the link itself
    # only knows whether it is currently reaching the aircraft.
    assert s["connected"] is True
    assert s["host"] == payload_server
    assert s["verdict"] == "PERSON"          # a detection outranks a gate hit
    assert s["scene"] == "flooded_areas"
    assert s["fix"] is True and s["sats"] == 7


def test_nearby_captures_merge_into_one_target(payload_server, tmp_path):
    db = tmp_path / "t.db"
    _run(payload_server, db)
    targets = Store(str(db)).all_targets()
    assert len(targets) == 3, "events 1 and 2 are metres apart and are one target"
    merged = [t for t in targets if t["n_passes"] == 2]
    assert len(merged) == 1
    # p_final takes the MAX, never the latest: a confident sighting is not
    # undone by a later glancing one of the same person.
    assert merged[0]["p_final"] == pytest.approx(0.81)
    assert merged[0]["class"] == "HIGH"


def test_capture_without_a_fix_gets_no_coordinates(payload_server, tmp_path):
    db = tmp_path / "t.db"
    _run(payload_server, db)
    unfixed = [t for t in Store(str(db)).all_targets() if t["lat"] is None]
    assert len(unfixed) == 1, "a find with no fix must still reach the queue"


def test_manual_datum_places_unfixed_captures_and_marks_them_uncertain(payload_server, tmp_path):
    db = tmp_path / "t.db"
    _run(payload_server, db, origin=(22.5000, 88.3000))
    placed = [t for t in Store(str(db)).all_targets()
              if t["lat"] == pytest.approx(22.5000)]
    assert len(placed) == 1
    # 25 m, not the 2.5 m a real fix claims: the ring on the map has to show
    # that this position was asserted by a person, not measured.
    assert placed[0]["pos_err_m"] == pytest.approx(25.0)


def test_events_are_not_ingested_twice(payload_server, tmp_path):
    db = tmp_path / "t.db"
    link = _run(payload_server, db, wait=3.0)
    time.sleep(0.6)                      # at least one more /events poll
    assert link.live()["ingested"] == len(EVENTS)


def test_merge_radius_matches_the_haversine_it_is_compared_against():
    # 8 m north of the origin must merge; 30 m must not. Deliberately off the
    # boundary itself -- a case that sits within a centimetre of the threshold
    # tests floating point, not the behaviour anyone cares about.
    assert _haversine_m(22.5726, 88.3639, 22.572672, 88.3639) < MERGE_RADIUS_M
    assert _haversine_m(22.5726, 88.3639, 22.57287, 88.3639) > MERGE_RADIUS_M


@pytest.mark.parametrize("conf,expected", [(0.9, "HIGH"), (0.6, "HIGH"),
                                           (0.45, "MEDIUM"), (0.2, "LOW")])
def test_confidence_bands(conf, expected):
    assert _klass(conf) == expected


def _wait_for(link, n, timeout=4.0):
    end = time.time() + timeout
    while time.time() < end and link.live().get("ingested", 0) < n:
        time.sleep(0.05)
    return link.live().get("ingested", 0)


def test_a_new_payload_session_clears_the_previous_mission(payload_server, tmp_path):
    """Power cycling the aircraft starts a new mission, not a continuation.

    Carrying the last sortie's targets forward would put old finds on the new
    flight's map -- the kind of mistake that sends a team to an empty building.
    """
    db = tmp_path / "t.db"
    link = PayloadLink(payload_server, store_factory=lambda: Store(str(db)),
                       media_root=None, events_s=0.3)
    link.start()
    try:
        _wait_for(link, len(EVENTS))
        first = Store(str(db)).all_targets()
        assert len(first) == 3
        assert link.live()["session"] == "ab"

        # The payload restarts: same kinds of events, a new token, ids from 1.
        LIVE["events"] = [dict(e, s="cd") for e in EVENTS[:2]]
        end = time.time() + 5.0
        while time.time() < end and link.live().get("session") != "cd":
            time.sleep(0.05)
        assert link.live()["session"] == "cd", "link never noticed the new session"
        _wait_for(link, 2)

        after = Store(str(db)).all_targets()
        # Events 1 and 2 are metres apart, so the new mission is ONE target --
        # not three carried over plus a new one.
        assert len(after) == 1, f"previous mission was not cleared: {after}"
        assert link.live()["ingested"] == 2
    finally:
        link.stop()


def test_a_link_drop_is_not_a_new_mission(payload_server, tmp_path):
    """A radio blip must never wipe the flight in progress.

    The purge keys on the session TOKEN changing, never on connectivity, so an
    aircraft passing behind a building keeps its targets.
    """
    db = tmp_path / "t.db"
    link = PayloadLink(payload_server, store_factory=lambda: Store(str(db)),
                       media_root=None, events_s=0.3)
    link.start()
    try:
        _wait_for(link, len(EVENTS))
        before = len(Store(str(db)).all_targets())
        assert before == 3

        # Same token, events keep arriving -- as after any reconnect.
        LIVE["events"] = list(EVENTS)
        time.sleep(1.0)
        assert len(Store(str(db)).all_targets()) == before
        assert link.live()["ingested"] == len(EVENTS)
    finally:
        link.stop()


def test_mission_clears_once_the_payload_has_been_off_long_enough(tmp_path):
    """A shutdown ends the mission; a blip does not.

    Evidence and the review queue sitting there populated against a dead
    payload is the stale-data case: it reads as live, and an operator cannot
    tell yesterday's bench run from this morning's.
    """
    import time as _t
    from saresq.dashboard import payload as P

    link = P.PayloadLink("127.0.0.1:1", lambda: Store(str(tmp_path / "m.db")),
                         media_root=str(tmp_path / "media"))
    link._session = "abc"
    link._seen.add(("abc", 1))

    # A short gap is a blip: nothing is cleared.
    link._last_ok = _t.time() - 2.0
    link._purge_if_payload_gone()
    assert link._session == "abc"
    assert link._seen

    # Beyond the threshold it is a shutdown, and the mission ends.
    link._last_ok = _t.time() - (P.PAYLOAD_GONE_S + 1.0)
    link._purge_if_payload_gone()
    assert link._session is None
    assert not link._seen


def test_a_payload_never_seen_is_not_a_payload_that_vanished(tmp_path):
    """Before the first successful poll there is no mission to clear, and
    purging then would wipe a store the operator deliberately loaded with
    --db before the aircraft was even switched on."""
    from saresq.dashboard import payload as P

    link = P.PayloadLink("127.0.0.1:1", lambda: Store(str(tmp_path / "m.db")),
                         media_root=str(tmp_path / "media"))
    link._last_ok = 0.0
    link._session = None
    link._purge_if_payload_gone()          # must not raise, must not purge
    assert link._session is None


def test_counting_people_needs_more_evidence_than_finding_one(tmp_path):
    """Presence and count are different claims and take different bars.

    Calibrated live: with one person in shot, every threshold up to 0.40
    reported two on some frames -- a phantom box at 0.433 beside the real
    detection. A number shown to an operator is a claim about how many people
    need rescuing, so it takes the strict bar; deciding somebody is there at
    all takes the permissive one, because a second look is the cheap error.
    """
    from saresq.dashboard import payload as P

    link = P.PayloadLink("127.0.0.1:1", lambda: Store(str(tmp_path / "m.db")))

    # One solid detection plus the phantom: exactly the measured case.
    link._ground_look = {"n": sum(1 for c in (0.60, 0.433)
                                  if c >= P.GROUND_COUNT_CONF),
                         "present": True, "p": 0.60, "ms": 90.0,
                         "t": __import__("time").time()}
    assert link._ground_look["n"] == 1        # the 0.433 box is not a person

    # A weak lone box: enough to look, not enough to count.
    link._ground_look = {"n": 0, "present": True, "p": 0.30, "ms": 90.0,
                         "t": __import__("time").time()}
    st = {"gate": {"fired": True}, "detect": {"n": 0, "rgb_blind": False},
          "motion": {"moved": False}, "thermal": {}, "gps": {}, "hazard": {}}
    out = link._shape(st)
    assert out["verdict"] == "PERSON"          # presence carried the ladder
    assert "1 person" not in (out["verdict_why"] or "")   # but named no count


def test_a_stale_ground_look_stops_counting(tmp_path):
    """A frozen second opinion must not keep asserting somebody who has
    walked out of shot."""
    import time as _t
    from saresq.dashboard import payload as P

    link = P.PayloadLink("127.0.0.1:1", lambda: Store(str(tmp_path / "m.db")))
    link._ground_look = {"n": 3, "present": True, "p": 0.9, "ms": 90.0,
                         "t": _t.time() - (P.GROUND_LOOK_TTL_S + 1.0)}
    st = {"gate": {"fired": True}, "detect": {"n": 0, "rgb_blind": True},
          "motion": {"moved": False}, "thermal": {}, "gps": {}, "hazard": {}}
    out = link._shape(st)
    assert out["n"] == 0
    assert out["verdict"] == "BODY_HEAT"
