"""Derive the Koopman sweep width W from measured detector recall (Section 12).

    python training/eval_sweep_width.py --pixels results/detector/pixels_on_target.csv \
        --gate-recall 0.82

``W`` is the single most load-bearing constant in the search model. Everything
the mission planner claims -- coverage C = W*L/A, probability of detection
POD = 1 - exp(-C), the adaptive-vs-lawnmower comparison in ``sim/`` -- scales
directly with it, and the number in use so far (14 m) was an assumption
inherited from the spec, never measured. This script replaces it with one
derived from the payload's own geometry and the detector's own measured recall.

**W is not "how wide the camera sees".** It is the integral of the lateral
range curve: W = the integral over lateral offset x of POD(x) dx. A sensor that
sees 20 m wide but only detects half of what is under it has a sweep width of
10 m. That definition is why W can be measured at all, and why quoting the
swath width as W would overstate coverage by ~50%.

**A result that surprised me, worth keeping.** The instinct is that targets at
the edge of the swath are harder -- longer slant range, oblique view -- so the
lateral range curve should fall off towards the edges. For a nadir-pointing
pinhole camera over flat ground it does not. A ground point at lateral offset x
images at u = f*x/h, so du/dx = f/h is *constant*: ground sample distance is
uniform across the entire swath, and the perspective foreshortening exactly
cancels the increased range. So the curve is flat, not domed, and the only
geometric falloff is at the very edge where a target is partially outside the
frame. That truncation is modelled below; it is small but real.

What this script does NOT model, and must not be read as including: lens
vignetting, off-axis MTF loss, thermal detector angular sensitivity, terrain
relief, or occlusion by rubble. Those all push W down. Treat the output as an
optimistic bound on the geometric contribution, gated by measured recall.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import yaml


def swath_width_m(fov_deg: float, altitude_m: float) -> float:
    """Ground width covered across-track by a nadir camera."""
    return 2.0 * altitude_m * math.tan(math.radians(fov_deg) / 2.0)


def edge_truncation_factor(swath_px: int, target_px: float) -> float:
    """Fraction of the swath where a whole target fits inside the frame.

    A target is only reliably detectable if it is not cut by the frame edge.
    Modelling the transition as linear over one target-width at each edge, the
    integral of the lateral range curve loses one target width in total.
    """
    if swath_px <= 0:
        return 0.0
    return max(0.0, 1.0 - target_px / swath_px)


def recall_at_size(rows: list[dict], variant: str, target_px: float) -> tuple[float, str]:
    """Measured recall from the pixels-on-target table, for the bin containing
    ``target_px`` at the native (scale 1.0) resolution."""
    for row in rows:
        if row["variant"] != variant or float(row["scale"]) != 1.0:
            continue
        lo_s, hi_s = row["bin_px"].split("-")
        lo = float(lo_s)
        hi = float("inf") if hi_s == "inf" else float(hi_s)
        if lo <= target_px < hi:
            return float(row["recall"]), row["bin_px"]
    raise SystemExit(
        f"no scale=1.0 row for variant {variant!r} covering {target_px:.1f} px; "
        f"run training/eval_pixels_on_target.py first"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--config", default=str(repo / "configs" / "pipeline.yaml"))
    ap.add_argument("--pixels", default=str(repo / "results" / "detector" / "pixels_on_target.csv"))
    ap.add_argument("--variant", default="p2")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gate-recall", type=float,
                     help="MEASURED thermal gate recall. Deliberately has no default: a "
                          "plausible-looking guess here silently sets the whole search "
                          "model's central constant.")
    src.add_argument("--gate-recall-json",
                     help="read it from make_fusion_dataset.py's sidecar instead")
    ap.add_argument("--person-length-m", type=float, default=1.70)
    ap.add_argument("--assumed-w", type=float, default=14.0, help="the value being replaced")
    ap.add_argument("--out", default=str(repo / "results" / "detector" / "sweep_width.json"))
    args = ap.parse_args()

    if args.gate_recall_json:
        sidecar = json.loads(pathlib.Path(args.gate_recall_json).read_text())
        if "gate_recall" not in sidecar:
            raise SystemExit(f"{args.gate_recall_json} has no 'gate_recall'; regenerate it "
                             f"with training/make_fusion_dataset.py")
        gate_recall = float(sidecar["gate_recall"])
        gate_source = f"measured, {args.gate_recall_json}"
    else:
        gate_recall = args.gate_recall
        gate_source = "supplied on the command line"

    cfg = yaml.safe_load(open(args.config))
    alt = float(cfg["mission"]["survey_alt_m"])
    speed = float(cfg["mission"]["survey_speed_mps"])
    thermal_fov = float(cfg["sensors"]["thermal"]["fov_deg"][0])
    rgb_fov = float(cfg["sensors"]["rgb"]["fov_deg"][0])
    thermal_px = int(cfg["sensors"]["thermal"]["shape"][1])
    rgb_px = int(cfg["sensors"]["rgb"]["capture"][0])
    focal_px = float(cfg["sensors"]["rgb"]["focal_px"])

    thermal_swath = swath_width_m(thermal_fov, alt)
    rgb_swath = swath_width_m(rgb_fov, alt)
    # The gate nominates every candidate, so nothing outside the THERMAL swath
    # can ever reach the detector. The binding swath is the narrower one.
    effective_swath = min(thermal_swath, rgb_swath)

    thermal_gsd = thermal_swath / thermal_px
    person_thermal_px = args.person_length_m / thermal_gsd
    rgb_gsd = alt / focal_px
    person_rgb_px = args.person_length_m / rgb_gsd

    rows = list(csv.DictReader(open(args.pixels)))
    det_recall, det_bin = recall_at_size(rows, args.variant, person_rgb_px)

    trunc = edge_truncation_factor(thermal_px, person_thermal_px)
    pod_interior = gate_recall * det_recall
    sweep_width = effective_swath * pod_interior * trunc

    lane = effective_swath  # planner's lane spacing, if lanes just abut
    area_m2 = 1.0e6  # 1 km^2 reference
    endurance_s = 20 * 60
    track_len = speed * endurance_s
    coverage = sweep_width * track_len / area_m2
    pod_mission = 1.0 - math.exp(-coverage)
    coverage_assumed = args.assumed_w * track_len / area_m2
    pod_assumed = 1.0 - math.exp(-coverage_assumed)

    result = {
        "measured": {
            "detector_variant": args.variant,
            "detector_recall": det_recall,
            "detector_recall_bin_px": det_bin,
            "gate_recall": gate_recall,
            "gate_recall_source": gate_source,
        },
        "geometry": {
            "altitude_m": alt,
            "thermal_swath_m": round(thermal_swath, 2),
            "rgb_swath_m": round(rgb_swath, 2),
            "effective_swath_m": round(effective_swath, 2),
            "thermal_gsd_m_per_px": round(thermal_gsd, 4),
            "person_thermal_px": round(person_thermal_px, 2),
            "rgb_gsd_mm_per_px": round(rgb_gsd * 1000, 2),
            "person_rgb_px": round(person_rgb_px, 1),
            "edge_truncation_factor": round(trunc, 4),
        },
        "sweep_width_m": round(sweep_width, 2),
        "assumed_sweep_width_m": args.assumed_w,
        "ratio_measured_to_assumed": round(sweep_width / args.assumed_w, 3),
        "mission_example": {
            "note": "20 min at survey speed over 1 km^2, lanes abutting",
            "track_length_m": round(track_len, 1),
            "lane_spacing_m": round(lane, 2),
            "coverage_measured": round(coverage, 3),
            "pod_measured": round(pod_mission, 3),
            "coverage_assumed": round(coverage_assumed, 3),
            "pod_assumed": round(pod_assumed, 3),
        },
        "excluded_from_model": [
            "lens vignetting", "off-axis MTF loss", "thermal angular sensitivity",
            "terrain relief", "occlusion by rubble",
        ],
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nW = {sweep_width:.2f} m (was assuming {args.assumed_w} m) -> "
          f"mission POD {pod_mission:.1%} vs {pod_assumed:.1%}\nwrote {out}")


if __name__ == "__main__":
    main()
