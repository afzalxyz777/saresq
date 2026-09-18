"""The live verdict ladder: what the payload believes, right now, in one word.

THE ORDERING IS A CLAIM ABOUT EVIDENCE, NOT ABOUT CONFIDENCE SCORES
Three facts can be established independently, and they are not equally strong:

  thermal   something is radiating like a body, at z sigma above the scene
  motion    it changed shape between frames -- it is ALIVE
  visual    a detector looked at a picture of it and said "person"

The temptation is to rank the visual detection highest, because it is the one
that produces a confident-looking number and a green box. That is backwards
here, and saresq/fuse/classes.py has said so since it was written: "an RGB
detection with no thermal support is downgraded to LOW regardless of its own
score". A stock COCO detector will happily box a mannequin, a poster, a coat
over a chair, a photograph on a wall -- and a corpse. None of those need a
rescue team.

A body-temperature source that MOVES is a different kind of claim. It is two
independent physical measurements agreeing, neither of which a picture of a
person can fake, and between them they establish the thing the mission actually
searches for: not a human shape, but a living human.

So LIVE_BODY outranks a bare visual detection. This is not a UI preference; it
is the system's own doctrine applied to the live path, which had drifted from
it.

THE ONE PLACE THIS COULD MISLEAD, STATED LOUDLY
Motion proves life. Its ABSENCE PROVES NOTHING. An unconscious casualty -- the
most medically urgent person on any site -- does not move. BODY_HEAT therefore
sits immediately below LIVE_BODY and must never be presented as a weak or
low-priority result: it is a body-temperature source that is not moving, and
the two reasons for that are "it is not a person" and "they are unconscious".
The console says exactly that rather than implying the first.

This ladder ranks CONFIDENCE THAT A LIVING HUMAN IS PRESENT. It is not a
medical triage order, and it is not the ledger's calibrated probability --
those are different questions asked elsewhere.
"""
from __future__ import annotations

#: Highest first. Index is the operator-attention rank.
LADDER = [
    "LIVE_PERSON",   # thermal + motion + visual: every branch agrees
    "LIVE_BODY",     # thermal + motion: alive, whatever it looks like
    "PERSON",        # visual, corroborated by thermal
    "BODY_HEAT",     # thermal only -- still, or the camera could not see
    "VISUAL_ONLY",   # visual with NO thermal support: treat with suspicion
    "HEAT",          # a warm anomaly that is not behaving like a body
    "CLEAR",
]
RANK = {name: i for i, name in enumerate(LADDER)}

#: What each verdict means, in the words the console shows an operator.
MEANING = {
    "LIVE_PERSON": "moving body-temperature source, confirmed by the camera",
    "LIVE_BODY": "body-temperature source that MOVED — alive",
    "PERSON": "camera confirmed a person, with thermal support",
    "BODY_HEAT": "body-temperature source, not moving — may be unconscious",
    "VISUAL_ONLY": "camera says person, but no thermal signature — verify",
    "HEAT": "warm anomaly, not behaving like a body",
    "CLEAR": "nothing above threshold",
}


def verdict(*, n_visual: int, gate_fired: bool, moved: bool,
            rgb_blind: bool = False) -> str:
    """One word for the current state.

    `gate_fired` is the thermal branch's own judgement that something is
    anomalously warm; `moved` is saresq.thermal.motion; `n_visual` is how many
    people the visible detector found; `rgb_blind` says the crops it was given
    had no usable light, so its silence carries no information.
    """
    has_visual = n_visual > 0

    if gate_fired and moved:
        return "LIVE_PERSON" if has_visual else "LIVE_BODY"
    if has_visual:
        # Doctrine: a person is a thermal object first. A visual detection with
        # no thermal signature behind it is the WEAKEST evidence this system
        # can hold, not the strongest, however confident the box looked.
        return "PERSON" if gate_fired else "VISUAL_ONLY"
    if gate_fired:
        # No motion and no visual. If the camera was blind its silence is not
        # evidence of absence, and either way a still body-temperature source
        # is exactly what an unconscious casualty looks like.
        return "BODY_HEAT" if rgb_blind else "HEAT"
    return "CLEAR"


def outranks(a: str, b: str) -> bool:
    """True when verdict `a` deserves an operator's attention before `b`."""
    return RANK.get(a, len(LADDER)) < RANK.get(b, len(LADDER))
