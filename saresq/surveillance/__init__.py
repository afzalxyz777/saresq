"""Radar-style track processing for the ground station.

The dashboard used to plot whatever row happened to be in `targets` at the
moment you refreshed. That is a database view, not a surveillance picture: it
cannot tell you that a contact is stale, cannot hold a track through a gap in
the data, and cannot distinguish "the aircraft is here" from "the aircraft was
here 40 seconds ago and this is where it probably is now".

This package borrows the machinery that terminal air-traffic radar has used for
decades -- alpha-beta smoothing, a validation gate, M-of-N track initiation and
explicit coasting -- so the operator sees the same distinctions a controller
sees. Track status flags follow EUROCONTROL ASTERIX Category 062 (SDPS Track
Messages) naming so the vocabulary is one a surveillance engineer recognises.

NOT to be confused with `saresq.track`, which is a completely different thing:
that one tracks *blobs across video frames* in image space (SORT, IoU
association, ego-motion compensation) and runs on the aircraft. This one tracks
*objects across the ground* in WGS-84 and runs on the ground station. They sit
at opposite ends of the pipeline and share no code.
"""
from saresq.surveillance.filters import AlphaBeta, gate_radius_m
from saresq.surveillance.tracker import Track, TrackState, Tracker, TrackerConfig

__all__ = [
    "AlphaBeta",
    "gate_radius_m",
    "Track",
    "TrackState",
    "Tracker",
    "TrackerConfig",
]
