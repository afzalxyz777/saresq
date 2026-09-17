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


@pytest.fixture()
def payload_server():
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep pytest output clean
            pass

        def do_GET(self):  # noqa: N802
            body = json.dumps(EVENTS if self.path == "/events" else STATS).encode()
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
