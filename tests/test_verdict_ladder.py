"""The verdict ladder, including the ordering claim that motivates it."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from saresq.fuse.verdict import outranks, phrase, verdict   # noqa: E402


def test_every_branch_agreeing_is_the_top():
    assert verdict(n_visual=1, gate_fired=True, moved=True) == "LIVE_PERSON"


def test_a_moving_body_outranks_a_bare_visual_detection():
    """The claim this module exists to make. A stock COCO detector will box a
    mannequin, a poster, a coat on a chair -- and a corpse. A body-temperature
    source that moves is two independent physical measurements agreeing."""
    moving = verdict(n_visual=0, gate_fired=True, moved=True)
    seen = verdict(n_visual=1, gate_fired=True, moved=False)
    assert moving == "LIVE_BODY" and seen == "PERSON"
    assert outranks(moving, seen)


def test_a_visual_detection_with_no_thermal_support_is_weakest():
    """saresq/fuse/classes.py: 'an RGB detection with no thermal support is
    downgraded to LOW regardless of its own score'."""
    v = verdict(n_visual=3, gate_fired=False, moved=False)
    assert v == "VISUAL_ONLY"
    assert outranks("BODY_HEAT", v)
    assert outranks("PERSON", v)


def test_body_heat_sits_directly_below_live_body():
    """Motion proves life; its absence proves nothing. An unconscious casualty
    does not move, so a still body-temperature source must not be demoted far."""
    assert outranks("LIVE_BODY", "BODY_HEAT")
    assert outranks("BODY_HEAT", "VISUAL_ONLY")
    assert outranks("BODY_HEAT", "HEAT")


def test_blind_camera_yields_body_heat_not_plain_heat():
    assert verdict(n_visual=0, gate_fired=True, moved=False,
                   rgb_blind=True) == "BODY_HEAT"
    assert verdict(n_visual=0, gate_fired=True, moved=False,
                   rgb_blind=False) == "HEAT"


def test_nothing_at_all():
    assert verdict(n_visual=0, gate_fired=False, moved=False) == "CLEAR"


def test_motion_wins_even_when_the_camera_is_blind():
    assert verdict(n_visual=0, gate_fired=True, moved=True,
                   rgb_blind=True) == "LIVE_BODY"


def test_ladder_is_a_total_order():
    from saresq.fuse.verdict import LADDER
    for i, a in enumerate(LADDER):
        for b in LADDER[i + 1:]:
            assert outranks(a, b) and not outranks(b, a)
def test_moving_body_with_a_blind_camera_reads_as_likely_survivors():
    """The night case this payload exists for must not report a MISSING
    count as an absent person. Thermal cannot resolve individuals, so no
    number is claimed -- but movement plus body heat is a conclusion."""
    said = phrase("LIVE_BODY", rgb_blind=True)
    assert "likely survivors" in said
    assert "camera blind" in said
    # It must never invent a count off the thermal branch.
    assert not any(ch.isdigit() for ch in said)


def test_a_camera_that_could_see_keeps_cautious_wording():
    """If the camera had light and still found nobody, that silence is
    evidence, and the wording must not upgrade itself past it."""
    said = phrase("LIVE_BODY", rgb_blind=False)
    assert "likely survivors" not in said
    assert "did not confirm" in said


def test_a_real_visual_count_is_reported_as_a_count():
    assert phrase("LIVE_PERSON", n_visual=1).startswith("1 person confirmed")
    assert phrase("LIVE_PERSON", n_visual=3).startswith("3 persons confirmed")
