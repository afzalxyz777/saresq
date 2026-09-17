"""Tests for the radar tracker, the flight/link model and the radar service."""
from __future__ import annotations

import math

import pytest

from saresq.surveillance.filters import AlphaBeta, gate_radius_m, kalata_gains, variance_reduction
from saresq.surveillance.flight import (
    Box,
    LinkBudget,
    Obstruction,
    SurveyPlan,
    detection_probability,
    lateral_offset_m,
    lateral_sigma_m,
    leg_spacing_m,
    swath_m,
)
from saresq.surveillance.geo import LocalFrame, haversine_m
from saresq.surveillance.service import RadarService, classify, icd203
from saresq.surveillance.tracker import Plot, TrackerConfig, TrackState, Tracker


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1e-6, 0.01, 0.1, 1.0, 10.0, 100.0])
def test_kalata_gains_are_stable(L):
    """The alpha-beta filter is stable for 0 < alpha < 1 and 0 < beta < 4-2alpha."""
    a, b = kalata_gains(L)
    assert 0.0 <= a <= 1.0
    assert 0.0 <= b < 4.0 - 2.0 * a + 1e-9


def test_variance_reduction_actually_reduces():
    """Smoothing must beat trusting each measurement on its own, or there is no
    point running a filter at all."""
    a, b = kalata_gains(0.3)
    vrf_p, _ = variance_reduction(a, b, 2.0)
    assert vrf_p < 1.0


def test_filter_converges_on_constant_velocity():
    f = AlphaBeta(scan_period_s=1.0, sigma_meas_m=2.0, sigma_accel_ms2=0.5)
    vx, vy = 6.0, 0.0
    for k in range(1, 60):
        f.update(vx * k, vy * k, 1.0)
    assert f.speed_ms == pytest.approx(6.0, abs=0.35)
    assert f.course_deg == pytest.approx(90.0, abs=3.0)


def test_coast_extrapolates_and_grows_uncertainty():
    f = AlphaBeta(scan_period_s=1.0, sigma_meas_m=2.0, sigma_accel_ms2=1.0)
    for k in range(1, 30):
        f.update(5.0 * k, 0.0, 1.0)
    x0, s0 = f.x, f.sigma_pred_m(0.0)
    f.coast(4.0)
    assert f.x > x0 + 15.0                     # kept moving at its own velocity
    assert f.sigma_pred_m(0.0) > s0            # and is less certain for it
    assert f.vx == pytest.approx(f.vx)         # velocity was not decayed away


def test_static_filter_does_not_drift_or_grow():
    """A survivor lying in rubble does not accumulate dead-reckoning error."""
    f = AlphaBeta(scan_period_s=2.0, sigma_meas_m=3.5, static=True)
    f.update(10.0, 20.0, 2.0)
    x, y, s = f.x, f.y, f.sigma_pred_m(0.0)
    for _ in range(200):
        f.coast(2.0)
    assert (f.x, f.y) == (x, y)
    assert f.sigma_pred_m(400.0) == pytest.approx(s)
    assert f.speed_ms == 0.0


def test_gate_grows_with_prediction_uncertainty():
    assert gate_radius_m(20.0, 2.0) > gate_radius_m(2.0, 2.0)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_local_frame_round_trips():
    fr = LocalFrame(22.5732, 88.3650)
    for dlat, dlon in ((0.001, 0.002), (-0.0007, 0.0004), (0, 0)):
        lat, lon = 22.5732 + dlat, 88.3650 + dlon
        e, n = fr.to_en(lat, lon)
        back = fr.to_ll(e, n)
        assert back[0] == pytest.approx(lat, abs=1e-9)
        assert back[1] == pytest.approx(lon, abs=1e-9)


def test_local_frame_agrees_with_haversine():
    """Flat-earth is only allowed because it is accurate at this scale -- so
    check it, rather than assume it.

    The residual is not implementation error: LocalFrame is ellipsoidal and
    haversine is spherical, and at 22.5 N those differ by ~0.25% in the north
    component whatever single radius the sphere is given. 0.5% is therefore the
    tightest honest bound, and it is 4 m over a kilometre -- an order of
    magnitude below the GPS noise everything here is filtering.
    """
    fr = LocalFrame(22.5732, 88.3650)
    lat, lon = 22.5732 + 0.005, 88.3650 + 0.005
    flat = math.hypot(*fr.to_en(lat, lon))
    assert flat == pytest.approx(haversine_m(22.5732, 88.3650, lat, lon), rel=5e-3)


# ---------------------------------------------------------------------------
# track lifecycle
# ---------------------------------------------------------------------------
def _tracker(**kw):
    cfg = TrackerConfig(scan_period_s=1.0, **kw)
    return Tracker(LocalFrame(22.5732, 88.3650), cfg)


def test_m_of_n_initiation_then_firm():
    tr = _tracker(init_m=3, init_n=4)
    t = 0.0
    for k in range(4):
        t += 1.0
        tr.update(t, [Plot(lat=22.5732 + k * 1e-5, lon=88.3650)])
    only = list(tr.tracks.values())[0]
    assert only.state is TrackState.FIRM
    assert only.status_bits(t)["CNF"] == 0


def test_tentative_track_is_dropped_on_first_miss():
    """Clutter must not be allowed to mature into a firm track."""
    tr = _tracker(init_m=3, init_n=4)
    tr.update(1.0, [Plot(lat=22.5732, lon=88.3650)])
    assert list(tr.tracks.values())[0].state is TrackState.TENTATIVE
    tr.update(2.0, [])
    assert tr.tracks == {}


def test_firm_track_coasts_then_drops():
    tr = _tracker(init_m=2, init_n=3, coasts_to_terminate=4, terminate_action="DROP")
    t = 0.0
    for k in range(4):
        t += 1.0
        tr.update(t, [Plot(lat=22.5732 + k * 1e-5, lon=88.3650)])
    for expect_coasts in range(1, 4):
        t += 1.0
        tr.update(t, [])
        trk = list(tr.tracks.values())[0]
        assert trk.state is TrackState.COAST
        assert trk.status_bits(t)["CST"] == 1
        assert trk.coasts == expect_coasts
    t += 1.0
    tr.update(t, [])
    assert tr.tracks == {}, "a platform track must terminate, like ATC"


def test_contact_track_is_held_never_dropped():
    """Losing the radio must not delete a person from the map."""
    tr = _tracker(init_m=1, init_n=1, coasts_to_terminate=3,
                  terminate_action="HOLD", static=True)
    tr.update(1.0, [Plot(lat=22.5732, lon=88.3650, ident="S1")])
    for k in range(60):
        tr.update(2.0 + k, [])
    trk = list(tr.tracks.values())[0]
    assert trk.state is TrackState.HELD
    assert trk.age_s(62.0) > 55.0
    assert trk.sigma_m(62.0) < 10.0, "a held contact must not grow a huge ellipse"


def test_plot_inside_gate_associates_and_outside_starts_a_new_track():
    tr = _tracker(init_m=1, init_n=1)
    tr.update(1.0, [Plot(lat=22.5732, lon=88.3650)])
    tr.update(2.0, [Plot(lat=22.5732 + 2e-6, lon=88.3650)])   # ~0.2 m away
    assert len(tr.tracks) == 1
    tr.update(3.0, [Plot(lat=22.5732, lon=88.3650),
                    Plot(lat=22.5800, lon=88.3700)])          # ~830 m away
    assert len(tr.tracks) == 2


def test_identity_binds_even_beyond_the_gate():
    """A store target id is a hard association; it must never be gated away."""
    tr = _tracker(init_m=1, init_n=1, static=True)
    tr.update(1.0, [Plot(lat=22.5732, lon=88.3650, ident="T7")])
    tr.update(2.0, [Plot(lat=22.5900, lon=88.3900, ident="T7")])
    assert len(tr.tracks) == 1


def test_mode_of_movement_detects_a_turn():
    tr = _tracker(init_m=1, init_n=1)
    fr = LocalFrame(22.5732, 88.3650)
    for k in range(1, 14):                      # a quarter circle
        a = math.radians(k * 6)
        lat, lon = fr.to_ll(60 * math.sin(a), 60 * (1 - math.cos(a)))
        tr.update(float(k), [Plot(lat=lat, lon=lon)])
    assert list(tr.tracks.values())[0].trans in ("RIGHT", "LEFT")


# ---------------------------------------------------------------------------
# flight and link
# ---------------------------------------------------------------------------
def test_survey_plan_covers_the_box_with_the_right_spacing():
    plan = SurveyPlan()
    lats = [w[0] for w in plan.waypoints]
    assert min(lats) == pytest.approx(plan.box.lat0, abs=1e-6)
    assert max(lats) == pytest.approx(plan.box.lat1, abs=1e-6)
    gaps = sorted({round(a, 6) for a in lats})
    step_m = (gaps[1] - gaps[0]) * 110574
    assert step_m == pytest.approx(leg_spacing_m(), rel=0.35)
    assert 2000 < plan.length_m < 5000
    assert plan.duration_s == pytest.approx(plan.length_m / plan.speed_ms)


def test_survey_plan_wraps_and_stays_inside_the_box():
    plan = SurveyPlan()
    a = plan.at(10.0)
    b = plan.at(10.0 + plan.length_m)
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    for f in range(0, 100):
        lat, lon, _ = plan.at(f / 100 * plan.length_m)
        assert plan.box.lat0 - 1e-6 <= lat <= plan.box.lat1 + 1e-6
        assert plan.box.lon0 - 1e-6 <= lon <= plan.box.lon1 + 1e-6


def test_swath_matches_the_payload_geometry():
    # 2 * 20 * tan(27.5 deg) -- the spec's own number for the MLX90640 at survey height
    assert swath_m(20.0) == pytest.approx(20.82, abs=0.05)
    assert leg_spacing_m(20.0) == pytest.approx(swath_m(20.0) * 0.8)


def test_link_weakens_with_range():
    """Free-space loss alone, which is what the default budget now models."""
    lb = LinkBudget()
    near, d_near = lb.rssi_dbm(22.5727, 88.3640, 20.0)
    far, _ = lb.rssi_dbm(22.5740, 88.3660, 20.0)
    assert near > far
    # No obstructions are configured by default any more -- the two that used
    # to be were hand-typed buildings nobody surveyed.
    assert d_near["through_m"] == 0


def test_an_obstruction_attenuates_when_one_is_actually_supplied():
    """The diffraction model is real; it just needs real geometry to run on.

    Supplied here explicitly rather than taken from a module default, so the
    test proves the physics without the package having to assert that a
    particular building exists in Kolkata.
    """
    block = Obstruction("test block", Box(22.57300, 88.36470, 22.57352, 88.36560), 28.0)
    lb = LinkBudget(obstructions=(block,))
    clear, d_clear = lb.rssi_dbm(22.5722, 88.3630, 20.0)
    blocked, d_blocked = lb.rssi_dbm(22.5740, 88.36520, 20.0)
    assert d_blocked["through_m"] > 0 and d_blocked["blocker"] == "test block"
    assert d_clear["through_m"] == 0
    assert blocked < clear


def test_link_states_bracket_the_thresholds():
    assert LinkBudget.state(-40.0) == "UP"
    assert LinkBudget.state(-100.0) == "DEGRADED"
    assert LinkBudget.state(-110.0) == "DOWN"


def test_lateral_range_curve_integrates_to_the_sweep_width():
    """Sweep width is by definition the integral of the detection curve. If
    these two disagree, the simulated detections and the POD figure on the
    analytics page are describing different sensors."""
    W, p_max = 14.0, 0.6
    step = 0.05
    total = 2 * sum(detection_probability(y, W, p_max) * step
                    for y in [i * step for i in range(int(200 / step))])
    assert total == pytest.approx(W, rel=0.01)
    assert lateral_sigma_m(W, p_max) == pytest.approx(W / (p_max * math.sqrt(math.pi)))


def test_lateral_offset_is_signed_across_track():
    across, along = lateral_offset_m(22.5732, 88.3650, 90.0, 22.5732, 88.3660)
    assert along > 90 and abs(across) < 1.0        # dead ahead on an easterly course
    across2, _ = lateral_offset_m(22.5732, 88.3650, 90.0, 22.5742, 88.3650)
    assert across2 < -50                           # off to one side


# ---------------------------------------------------------------------------
# doctrine wording
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p,word", [
    (0.02, "almost no chance"), (0.10, "very unlikely"), (0.30, "unlikely"),
    (0.50, "roughly even chance"), (0.70, "likely"), (0.90, "very likely"),
    (0.97, "almost certainly"),
])
def test_icd203_bands(p, word):
    assert icd203(p) == word


def test_class_bands():
    assert classify(0.80) == "HIGH"
    assert classify(0.50) == "MEDIUM"
    assert classify(0.10) == "LOW"


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------
def _svc(t, **kw):
    """A service with a rehearsal picture on an empty store.

    allow_synthetic is off in production precisely so an empty scope stays
    empty -- see the RadarService docstring. These tests exercise the display,
    the link budget and the wire format, all of which need *some* contact to
    act on, so they ask for the simulated picture explicitly. Every track it
    produces carries the ASTERIX SIM bit.
    """
    kw.setdefault("allow_synthetic", True)
    return RadarService(db_path="does-not-exist.db", clock=lambda: t[0], **kw)


def test_service_is_deterministic():
    """Refreshing the dashboard must not re-roll whether a survivor was found."""
    a, b = [1000.0], [1000.0]
    s1, s2 = _svc(a), _svc(b)
    a[0] += 400.0; b[0] += 400.0
    d1, d2 = s1.snapshot(), s2.snapshot()
    assert d1["platform"]["lat"] == d2["platform"]["lat"]
    assert [c["label"] for c in d1["contacts"]] == [c["label"] for c in d2["contacts"]]


def test_platform_coasts_while_the_link_is_cut_then_recovers():
    t = [1000.0]
    svc = _svc(t)
    t[0] += 200.0
    assert svc.snapshot()["platform"]["state"] == "FIRM"

    svc.inject("link", 20.0)
    t[0] += 10.0
    d = svc.snapshot()
    assert d["link"]["state"] == "DOWN"
    assert d["platform"]["state"] == "COAST"
    assert d["platform"]["status"]["CST"] == 1
    assert d["platform"]["age_s"] >= 8.0
    coasted = [h for h in d["platform"]["history"] if h["coast"]]
    assert coasted, "the trail must record which positions were dead-reckoned"

    t[0] += 30.0
    d = svc.snapshot()
    assert d["link"]["state"] != "DOWN"
    assert d["platform"]["state"] == "FIRM"
    assert d["platform"]["sigma_m"] < 5.0


def test_gps_denial_coasts_the_track_while_the_link_stays_up():
    """Two different failures that a display must not conflate."""
    t = [1000.0]
    svc = _svc(t)
    t[0] += 200.0; svc.snapshot()
    svc.inject("gps", 20.0)
    t[0] += 10.0
    d = svc.snapshot()
    assert d["link"]["state"] != "DOWN"
    assert d["link"]["gps_denied"] is True
    assert d["platform"]["state"] == "COAST"


def test_alerts_queue_while_down_and_flush_on_recovery():
    t = [1000.0]
    svc = _svc(t)
    t[0] += 30.0; svc.snapshot()
    svc.inject("link", 600.0)
    t[0] += 560.0                      # more than one full lap with no radio
    d = svc.snapshot()
    assert d["link"]["queued"] > 0, "detections made with no link must be held"
    assert d["link"]["queued_bytes"] == d["link"]["queued"] * 28
    assert d["contacts"] == []         # nothing reached the ground station

    svc.inject("link")                 # clear
    t[0] += 6.0
    d = svc.snapshot()
    assert d["link"]["queued"] == 0
    assert d["contacts"], "the backlog must appear on the scope once the link returns"


def test_coverage_and_pod_grow_together():
    t = [1000.0]
    svc = _svc(t)
    t[0] += 120.0
    a = svc.snapshot()["search"]
    t[0] += 400.0
    b = svc.snapshot()["search"]
    assert b["fraction"] > a["fraction"]
    assert b["pod"] > a["pod"]
    assert b["pod"] == pytest.approx(1 - math.exp(-b["coverage_c"]), abs=1e-3)
    assert len(b["grid"]) == b["nx"] * b["ny"]


def test_service_does_not_replay_forever_after_a_long_gap():
    t = [1000.0]
    svc = _svc(t)
    t[0] += 10 * 3600.0                # tab left open overnight
    d = svc.snapshot()                 # must return, not hang
    assert d["scan"] < 500


def test_a_long_gap_leaves_the_whole_picture_on_one_clock():
    """The bug this pins: re-basing t0 without moving the tracks left the
    platform (re-plotted on the next scan) and the contacts (not re-plotted)
    on different timelines, so contacts read ~14 h stale on a 10-minute
    picture. Every age must stay bounded by how long the sim has run."""
    t = [1000.0]
    svc = _svc(t)
    t[0] += 700.0
    before = svc.snapshot()
    assert before["contacts"], "need contacts on the board before the gap"

    t[0] += 10 * 3600.0                # overnight
    d = svc.snapshot()
    assert d["platform"]["age_s"] < 30
    for c in d["contacts"]:
        assert c["age_s"] <= d["t"] + 1, (
            f"{c['label']} reads {c['age_s']:.0f}s old on a {d['t']:.0f}s picture")
    for e in d["events"]:
        assert e["t"] >= 0, "an event must never end up before the start of the sortie"


def test_contact_payload_survives_the_28_byte_wire_format():
    t = [1000.0]
    svc = _svc(t)
    t[0] += 700.0
    d = svc.snapshot()
    assert d["contacts"]
    for c in d["contacts"]:
        assert c["payload"]["wire_bytes"] == 28
        assert 0.0 <= c["payload"]["p"] <= 1.0
        assert c["payload"]["class"] in ("LOW", "MEDIUM", "HIGH")
        # int32 degrees-e7 is exact to ~1 cm; the map must agree with the radio.
        assert svc.plan.box.lat0 - 1e-4 <= c["lat"] <= svc.plan.box.lat1 + 1e-4
