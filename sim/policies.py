"""Lawnmower flight geometry, battery model, and the three policies (Section 15.2/15.3).

This is a coarse discrete-event approximation of a flight, not a physically
simulated trajectory: each target is assigned an encounter time along the
sweep, evidence is drawn once per pass, and the ledger (saresq.fuse.ledger,
used completely unchanged) turns passes into a decision. The acceptance bar
for this module is comparative behaviour between policies (Section 19.1,
task 0.11), not flight-dynamics fidelity. Constants below are labelled where
they are a deliberate simplification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from saresq.fuse.ledger import Target
from sim.sensing import sense_pass
from sim.world import World

THERMAL_FOV_DEG = (55.0, 35.0)  # across-track, along-track (Section 2.2)

BATTERY_CAPACITY = 100.0
# Calibrated so a full 200x150 m survey at 12 m (the most energy-hungry
# altitude) uses ~90% of capacity, leaving the fixed-lawnmower-low control
# on the edge of running dry (Section 15.3's stated hypothesis) while a
# 20 m survey leaves comfortable margin for several adaptive interruptions.
COST_PER_METRE = 0.035
COST_PER_SECOND_HOVER = 0.3
# Approximates "fly from the interruption point to the target and back" for
# a target already near the current survey line; not a real path planner.
INTERRUPTION_TRANSIT_COST = 1.0

LEDGER_DEFAULTS = dict(pi_0=0.20, w_repeat=0.7, confirm_p=0.90, reject_p=0.10, max_passes=3)
ALT_BANDS_M = (25.0, 15.0)  # band 2: >25, band 1: 15-25, band 0: <15


def thermal_footprint(altitude_m: float, fov_deg: tuple[float, float] = THERMAL_FOV_DEG) -> tuple[float, float]:
    across = 2 * altitude_m * math.tan(math.radians(fov_deg[0] / 2))
    along = 2 * altitude_m * math.tan(math.radians(fov_deg[1] / 2))
    return across, along


def alt_band(altitude_m: float, bands: tuple[float, float] = ALT_BANDS_M) -> int:
    hi, lo = bands
    if altitude_m > hi:
        return 2
    if altitude_m > lo:
        return 1
    return 0


@dataclass
class TargetOutcome:
    is_distractor: bool
    encounter_time_s: float
    decision: str
    candidate_ever: bool
    reached: bool


@dataclass
class TrialResult:
    policy: str
    area_covered_m2: float
    survivors_confirmed: int
    distractors_confirmed: int
    survivors_missed: int
    time_to_first_confirmation_s: float  # inf if none confirmed
    time_to_confirm_all_s: float         # inf if not all confirmed
    battery_remaining_pct: float
    n_survivors: int
    n_distractors: int


def _lawnmower_plan(world: World, altitude_m: float, speed_mps: float):
    swath, _ = thermal_footprint(altitude_m)
    n_lines = max(1, math.ceil(world.height / swath))
    path_length = n_lines * world.width + max(n_lines - 1, 0) * swath
    return swath, n_lines, path_length


def _encounter_time(x: float, y: float, n_lines: int, world: World, swath: float, speed_mps: float) -> float:
    line_idx = min(int(y // swath), n_lines - 1)
    # Boustrophedon: every other line is traversed right-to-left.
    along_line = (world.width - x) if (line_idx % 2 == 1) else x
    distance = line_idx * (world.width + swath) + along_line
    return distance / speed_mps


def run_trial(
    rng: np.random.Generator,
    world: World,
    policy: str,
    survey_alt_m: float = 20.0,
    survey_speed_mps: float = 5.0,
    reobserve_alts_m: tuple[float, ...] = (12.0, 8.0),
    orbit_s: float = 8.0,
    sensing_overrides: dict | None = None,
) -> TrialResult:
    """policy in {'lawnmower', 'adaptive', 'lawnmower_low'}."""
    overrides = sensing_overrides or {}
    alt = survey_alt_m if policy != "lawnmower_low" else reobserve_alts_m[0]
    adaptive = policy == "adaptive"

    swath, n_lines, path_length = _lawnmower_plan(world, alt, survey_speed_mps)
    full_area = world.width * world.height

    targets = [(p, False) for p in world.survivors] + [(d, True) for d in world.distractors]
    encounters = []
    for obj, is_distractor in targets:
        t = _encounter_time(obj.x, obj.y, n_lines, world, swath, survey_speed_mps)
        encounters.append((t, obj, is_distractor))
    encounters.sort(key=lambda e: e[0])

    outcomes: list[TargetOutcome] = []
    # Battery is spent in chronological order on BOTH base flight and
    # interruption orbits, so a heavy reobserve schedule can ground the
    # aircraft before the survey itself is finished, not just after.
    cumulative_energy = 0.0
    base_distance_paid = 0.0
    base_distance_at_death = path_length  # unless the loop below says otherwise

    for idx, (t, obj, is_distractor) in enumerate(encounters):
        base_distance_now = t * survey_speed_mps
        cumulative_energy += (base_distance_now - base_distance_paid) * COST_PER_METRE
        base_distance_paid = base_distance_now

        if cumulative_energy > BATTERY_CAPACITY:
            base_distance_at_death = base_distance_paid
            for t2, _, is_distractor2 in encounters[idx:]:
                outcomes.append(TargetOutcome(is_distractor2, t2, "UNREACHED", False, False))
            break

        concealment = None if is_distractor else obj.concealment
        ledger_target = Target(lat=obj.x, lon=obj.y, **LEDGER_DEFAULTS)
        candidate_ever = False
        current_alt = alt
        cur_time = t

        # No blob and no RGB box means no candidate region and no track: the
        # ledger is never instantiated for real evidence, so there is nothing
        # to decide on. (Evaluating Target.decision() on zero passes would
        # wrongly return REOBSERVE_LOWER, since p_final defaults to pi_0.)
        pattern, p_k = sense_pass(rng, current_alt, concealment, world.ambient_band, is_distractor, **overrides)
        if pattern == "none":
            outcomes.append(TargetOutcome(is_distractor, cur_time, "NO_CANDIDATE", False, True))
            continue

        candidate_ever = True
        ledger_target.add_pass(alt_band(current_alt), p_k)
        decision = ledger_target.decision()

        if adaptive and decision == "REOBSERVE_LOWER":
            for reobs_alt in reobserve_alts_m:
                orbit_cost = INTERRUPTION_TRANSIT_COST + orbit_s * COST_PER_SECOND_HOVER
                if cumulative_energy + orbit_cost > BATTERY_CAPACITY:
                    break  # can't afford another orbit; keep the current decision as final
                cumulative_energy += orbit_cost
                cur_time += orbit_s
                pattern, p_k = sense_pass(rng, reobs_alt, concealment, world.ambient_band, is_distractor, **overrides)
                if pattern != "none":
                    ledger_target.add_pass(alt_band(reobs_alt), p_k)
                    decision = ledger_target.decision()
                if decision in ("CONFIRM", "REJECT", "LOG_AND_RESUME"):
                    break

        outcomes.append(TargetOutcome(is_distractor, cur_time, decision, candidate_ever, True))

    battery_remaining_pct = max(0.0, (BATTERY_CAPACITY - cumulative_energy) / BATTERY_CAPACITY * 100.0)
    area_covered = min(1.0, base_distance_at_death / path_length) * full_area

    survivor_outcomes = [o for o in outcomes if not o.is_distractor]
    distractor_outcomes = [o for o in outcomes if o.is_distractor]

    survivors_confirmed = sum(1 for o in survivor_outcomes if o.decision == "CONFIRM")
    distractors_confirmed = sum(1 for o in distractor_outcomes if o.decision == "CONFIRM")
    survivors_missed = sum(1 for o in survivor_outcomes if not o.candidate_ever)

    confirm_times = sorted(o.encounter_time_s for o in survivor_outcomes if o.decision == "CONFIRM")
    time_to_first = confirm_times[0] if confirm_times else math.inf
    time_to_all = confirm_times[-1] if len(confirm_times) == len(survivor_outcomes) and confirm_times else math.inf

    return TrialResult(
        policy=policy,
        area_covered_m2=area_covered,
        survivors_confirmed=survivors_confirmed,
        distractors_confirmed=distractors_confirmed,
        survivors_missed=survivors_missed,
        time_to_first_confirmation_s=time_to_first,
        time_to_confirm_all_s=time_to_all,
        battery_remaining_pct=battery_remaining_pct,
        n_survivors=len(survivor_outcomes),
        n_distractors=len(distractor_outcomes),
    )
