"""The live-contact projection.

These guard the two things that would be invisible if they broke: the geometry
(a blob at the centre of the array is directly under the aircraft, one at the
edge is off by the half-FOV), and the honesty flags (a contact whose heading
was guessed must SAY it was guessed).
"""
import math

import pytest

from saresq.dashboard.contacts import (DEFAULT_AGL_M, focal_px,
                                       project_contacts)

LAT, LON = 22.57323, 88.36497


def blob(row, col, **kw):
    b = {"row": row, "col": col, "z": 4.0, "T": 34.0, "px": 9}
    b.update(kw)
    return b


def test_no_datum_invents_nothing():
    """With no position there is no contact. Not a guess, not the origin."""
    assert project_contacts([blob(12, 16)], lat=None, lon=None) == []
    assert project_contacts([], lat=LAT, lon=LON) == []


def test_centre_blob_sits_under_the_aircraft():
    # The array centre is (row 12, col 16); a blob there is nadir, so its
    # ground position is the aircraft's own to within a metre.
    c = project_contacts([blob(11.5, 15.5)], lat=LAT, lon=LON, agl_m=20.0)[0]
    assert c["range_m"] < 0.05


def test_edge_blob_is_off_by_the_half_fov():
    # Bottom edge of the array, along-track: 35 deg FOV, so the half-angle is
    # 17.5 deg and the ground offset is h*tan(17.5) = 6.31 m at 20 m.
    c = project_contacts([blob(23.5, 15.5)], lat=LAT, lon=LON, agl_m=20.0)[0]
    assert c["range_m"] == pytest.approx(20.0 * math.tan(math.radians(17.5)),
                                         rel=0.02)


def test_range_scales_with_height():
    """Range is linear in AGL -- it is one angle and one height, nothing else."""
    a = project_contacts([blob(20, 20)], lat=LAT, lon=LON, agl_m=10.0)[0]
    b = project_contacts([blob(20, 20)], lat=LAT, lon=LON, agl_m=20.0)[0]
    assert b["range_m"] == pytest.approx(2.0 * a["range_m"], rel=1e-6)
    assert b["bearing_deg"] == pytest.approx(a["bearing_deg"], abs=1e-9)


def test_assumptions_are_declared():
    """The flags are the whole point: a guessed bearing must not look surveyed."""
    c = project_contacts([blob(20, 20)], lat=LAT, lon=LON)[0]
    assert "heading" in c["assumed"]      # no magnetometer on this payload
    assert "altitude" in c["assumed"]     # no AGL given, so DEFAULT_AGL_M
    c2 = project_contacts([blob(20, 20)], lat=LAT, lon=LON,
                          agl_m=3.0, yaw_deg=90.0)[0]
    assert c2["assumed"] == []


def test_manual_datum_is_flagged_and_widens_the_error():
    gps = project_contacts([blob(20, 20)], lat=LAT, lon=LON, agl_m=5.0,
                           yaw_deg=0.0, hdop=1.0)[0]
    man = project_contacts([blob(20, 20)], lat=LAT, lon=LON, agl_m=5.0,
                           yaw_deg=0.0, pos_source="manual")[0]
    assert "datum" in man["assumed"] and "datum" not in gps["assumed"]
    assert man["err_m"] > gps["err_m"] * 3


def test_heading_rotates_the_bearing():
    """Yaw turns the whole picture with the aircraft, and nothing else moves."""
    n = project_contacts([blob(23, 15.5)], lat=LAT, lon=LON,
                         agl_m=20.0, yaw_deg=0.0)[0]
    e = project_contacts([blob(23, 15.5)], lat=LAT, lon=LON,
                         agl_m=20.0, yaw_deg=90.0)[0]
    assert e["range_m"] == pytest.approx(n["range_m"], rel=1e-9)
    assert (e["bearing_deg"] - n["bearing_deg"]) % 360 == pytest.approx(90.0,
                                                                       abs=1e-6)


def test_gsd_matches_the_spec_sheet():
    """2h tan(FOV/2) / N -- Equation (1) in the paper, 55 deg over 32 px."""
    c = project_contacts([blob(12, 16)], lat=LAT, lon=LON, agl_m=20.0)[0]
    assert c["gsd_m"] == pytest.approx(
        2 * 20.0 * math.tan(math.radians(27.5)) / 32, rel=1e-9)


def test_error_never_undercuts_the_fix():
    """A contact can never be better located than the aircraft carrying it."""
    for hdop in (0.8, 1.0, 2.5, 6.0):
        c = project_contacts([blob(12, 16)], lat=LAT, lon=LON, agl_m=20.0,
                             yaw_deg=0.0, hdop=hdop)[0]
        assert c["err_m"] >= max(1.0, hdop) * 2.5


def test_focal_px_round_trips_the_fov():
    f = focal_px(55.0, 32)
    assert math.degrees(2 * math.atan((32 / 2) / f)) == pytest.approx(55.0)


def test_default_agl_is_the_survey_altitude():
    assert DEFAULT_AGL_M == 20.0


def test_relative_error_is_tighter_than_absolute_and_ignores_the_datum():
    """The map draws rel_err_m; drawing err_m would double-count the aircraft.

    A datum error moves the drone and the contact together, and the payload
    glyph already carries a ring for it. Two rings for one error is how a
    contact 20 cm from the aircraft ends up inside a 25 m circle.
    """
    gps = project_contacts([blob(20, 20, px=16)], lat=LAT, lon=LON, agl_m=20.0,
                           yaw_deg=0.0, hdop=1.0)[0]
    man = project_contacts([blob(20, 20, px=16)], lat=LAT, lon=LON, agl_m=20.0,
                           yaw_deg=0.0, pos_source="manual")[0]
    # relative is a property of the sensor and the height, not of the datum
    assert gps["rel_err_m"] == pytest.approx(man["rel_err_m"], rel=1e-9)
    # ...and it is always the smaller of the two
    assert gps["rel_err_m"] < gps["err_m"]
    assert man["rel_err_m"] < man["err_m"] / 10
