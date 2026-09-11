import numpy as np
import pytest

from saresq.store.db import Store
from saresq.store.media import MediaStore
from saresq.sync.agent import SyncAgent
from saresq.sync.alerts import ALERT_BYTES, Alert, alert_from_target, pack_alert, unpack_alert
from saresq.sync.link import Receiver, SimulatedLink, TelemetryLink, WifiLink


@pytest.fixture()
def rig(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    ms = MediaStore(tmp_path / "media", store)
    yield store, ms
    store.close()


# ---------------------------------------------------------------- alerts
def test_alert_is_28_bytes_and_survives_a_round_trip():
    a = Alert(target_id=7, t_s=1_757_000_000, lat=22.5731530, lon=88.3642300,
              p_final=0.96, class_="HIGH", decision="CONFIRM", n_passes=2, alt_m=10.0)
    buf = pack_alert(a)
    assert len(buf) == ALERT_BYTES == 28

    r = unpack_alert(buf)
    # degrees-e7 is exact to ~1 cm, well inside the 2.5 m geolocation CEP.
    assert abs(r.lat - a.lat) * 111_320 < 0.02
    assert abs(r.lon - a.lon) * 102_796 < 0.02
    assert r.class_ == "HIGH" and r.decision == "CONFIRM" and r.n_passes == 2
    assert r.alt_m == pytest.approx(10.0)


def test_alert_rejects_a_corrupted_packet():
    a = Alert(1, 0, 22.5, 88.3, 0.5, "LOW", "REJECT", 1, 20.0)
    buf = bytearray(pack_alert(a))
    buf[10] ^= 0xFF                      # flip a bit in the position field
    with pytest.raises(ValueError, match="CRC"):
        unpack_alert(bytes(buf))


def test_alert_fits_a_sik_radio_budget():
    """28 bytes over a ~16 kbps link is well under a tenth of a second, which
    is the whole reason alerts jump the queue ahead of imagery."""
    link = TelemetryLink()
    assert link.seconds_for(ALERT_BYTES + 1) < 0.02


# ---------------------------------------------------------------- ordering
def test_alerts_are_sent_before_any_imagery(rig):
    store, ms = rig
    tid = store.insert_target(first_seen_ns=1, last_seen_ns=2, lat=22.57, lon=88.36,
                              p_final=0.96, class_="HIGH", n_passes=2, decision="CONFIRM")
    ms.put_bytes(b"A" * 4000, "rgb_crop", target_id=tid, priority=0.96)
    store.insert_alert(tid, 3, pack_alert(alert_from_target(store.get_target(tid), t_s=3)))

    link = WifiLink()
    rep = SyncAgent(store, ms, link).pump(budget_s=10.0)

    assert rep.alerts_sent == 1 and rep.media_completed == 1
    # The alert frame is the very first thing the receiver saw.
    assert link.receiver.alerts, "alert should have been delivered"
    assert unpack_alert(link.receiver.alerts[0]).target_id == tid


# ---------------------------------------------------------------- resume
def test_transfer_resumes_at_the_exact_byte_after_a_link_drop(rig):
    store, ms = rig
    payload = bytes(np.random.default_rng(1).integers(0, 256, 40_000, dtype=np.uint8))
    mid = ms.put_bytes(payload, "clip", priority=0.9)

    receiver = Receiver()
    # Same receiver across both links: the ground station does not forget what
    # it already has just because the aircraft changed radios.
    link = SimulatedLink(goodput_bps=1_000_000, mtu=1024, down_after_bytes=12_000, receiver=receiver)
    agent = SyncAgent(store, ms, link)

    rep1 = agent.pump(budget_s=60.0)
    assert rep1.stopped == "link_down"
    assert rep1.media_completed == 0 and rep1.media_partial == 1
    progress = store.get_media(mid)["sent_bytes"]
    assert 0 < progress < len(payload)

    link.bring_up()
    rep2 = agent.pump(budget_s=60.0)
    assert rep2.media_completed == 1
    assert store.get_media(mid)["synced_ns"] is not None

    assert receiver.assembled(mid) == payload, "resumed transfer must be byte-exact"
    # The whole point of resuming: we did not pay for those bytes twice.
    assert link.stats.sent_bytes < len(payload) * 1.5


def test_budget_stops_cleanly_and_loses_nothing(rig):
    store, ms = rig
    payload = b"Z" * 30_000
    mid = ms.put_bytes(payload, "clip", priority=0.5)

    link = SimulatedLink(goodput_bps=100_000, mtu=1024)
    agent = SyncAgent(store, ms, link)

    rep = agent.pump(budget_s=0.5)          # ~6 KB of link time
    assert rep.stopped == "budget"
    assert store.get_media(mid)["synced_ns"] is None

    for _ in range(20):
        if agent.pump(budget_s=0.5).stopped == "empty":
            break
    assert store.get_media(mid)["synced_ns"] is not None
    assert link.receiver.assembled(mid) == payload


def test_agent_reports_link_down_without_touching_the_queue(rig):
    store, ms = rig
    ms.put_bytes(b"x" * 100, "thumb", priority=1.0)
    link = SimulatedLink()
    link.bring_down()

    rep = SyncAgent(store, ms, link).pump(budget_s=5.0)
    assert rep.stopped == "link_down" and not rep.did_work
    assert len(store.pending_media()) == 1, "nothing may be marked sent"


def test_missing_blob_does_not_stall_the_queue(rig):
    store, ms = rig
    orphan = ms.put_bytes(b"gone", "thumb", priority=1.0)
    good = ms.put_bytes(b"here" * 50, "rgb_crop", priority=0.9)
    ms.path_for(orphan).unlink()

    rep = SyncAgent(store, ms, WifiLink()).pump(budget_s=10.0)
    assert rep.media_completed == 1
    assert store.get_media(good)["synced_ns"] is not None
    assert store.get_media(orphan)["synced_ns"] is None


def test_backlog_reports_what_is_still_on_the_aircraft(rig):
    store, ms = rig
    ms.put_bytes(b"a" * 1000, "thumb", priority=0.9)
    ms.put_bytes(b"b" * 150_000, "clip", priority=0.9)

    b = SyncAgent(store, ms, TelemetryLink()).backlog()
    assert b["media_items"] == 2
    assert b["by_kind"] == {"thumb": 1, "clip": 1}
    assert b["bytes_remaining"] == 151_000
    # 151 KB over a SiK radio is over a minute -- the number that justifies
    # shipping thumbnails first rather than clips.
    assert b["seconds_at_link_rate"] > 60
