"""Put the gate's live thermal blobs on the map, next to the aircraft.

WHY THIS EXISTS
The map has always been able to draw TARGETS -- rows the payload uploaded,
clustered by the ledger, carrying a fused probability. Those only appear after
evidence has been ingested, which on a bench with no fix and no uplink is
never. Meanwhile the gate is firing at 5 sigma on a person three metres away
and the map shows one aircraft glyph and nothing else.

A CONTACT is the other half: where the thing the gate is looking at RIGHT NOW
sits on the ground. It is not a target. It has no fused probability, no pass
count and no ledger entry; it lives for as long as the blob does, and it
disappears when the blob does. The map draws the two differently for exactly
that reason.

WHAT IS MEASURED AND WHAT IS ASSUMED
This is the whole difficulty, so it is made explicit in the output rather than
buried in a comment:

  RANGE from the aircraft is real. It follows from the blob's angular offset
  in the array and the height above ground, and nothing else.

  BEARING is only real if the aircraft's heading is known. The payload has no
  magnetometer and the NEO-6M reports no course while stationary, so on the
  bench the heading is an assumption, and every contact derived from it is
  drawn as a relative bearing and labelled.

  HEIGHT ABOVE GROUND is assumed unless someone measures it. GNSS altitude is
  height above the ellipsoid, not above the roof the casualty is lying on, and
  subtracting one from the other needs a terrain model this payload does not
  carry.

Assumptions are returned in `assumed` so the caller can draw them differently
and say which ones applied. A contact whose bearing is assumed is still worth
drawing -- the operator learns "something at body temperature is 8 m from the
aircraft", which is a genuinely useful thing to know -- but it must never be
drawn as though the payload surveyed it.
"""
from __future__ import annotations

import math

from saresq.geo.project import pixel_to_latlon

#: MLX90640 across-track x along-track field of view, and the array shape.
#: These mirror sensors.thermal in configs/pipeline.yaml; they are passed in by
#: the caller rather than read here so this module stays a pure function.
THERMAL_FOV_DEG = (55.0, 35.0)
THERMAL_SHAPE = (24, 32)          # rows, cols

#: Height above ground when nobody has said otherwise. mission.survey_alt_m.
#: The map lets an operator change it, because a payload on a table in a
#: college room is one metre up, not twenty, and pinning contacts twenty metres
#: out from a drone that is sitting still is a worse lie than admitting the
#: number was guessed.
DEFAULT_AGL_M = 20.0

#: Contribution of the receiver to the position of a contact. The aircraft's
#: own fix is the dominant term and this is its nominal UERE per unit HDOP.
UERE_M = 2.5


def focal_px(fov_deg: float, n_px: float) -> float:
    """Pinhole focal length in pixels for a given field of view."""
    return (n_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def project_contacts(
    blobs,
    *,
    lat: float | None,
    lon: float | None,
    agl_m: float | None = None,
    yaw_deg: float | None = None,
    hdop: float | None = None,
    fov_deg=THERMAL_FOV_DEG,
    shape=THERMAL_SHAPE,
    pos_source: str = "gps",
) -> list[dict]:
    """Ground positions for the gate's current blobs.

    `blobs` are the dicts gate_stats() produces: row/col in thermal pixels,
    plus z, T and px. Returns one dict per blob, or [] when there is no datum
    to hang them off -- inventing a position for a contact is precisely the
    failure this project refuses to make elsewhere, and it would be no more
    acceptable here.
    """
    if lat is None or lon is None or not blobs:
        return []

    assumed = []
    if agl_m is None:
        agl_m = DEFAULT_AGL_M
        assumed.append("altitude")
    if yaw_deg is None:
        yaw_deg = 0.0
        assumed.append("heading")
    if pos_source != "gps":
        # The aircraft itself is not where it says it is; everything hung off
        # it inherits that. Named separately from the two above because it is
        # the operator's own datum, not a missing sensor.
        assumed.append("datum")

    rows, cols = shape
    fx = focal_px(fov_deg[0], cols)
    fy = focal_px(fov_deg[1], rows)

    # pixel_to_latlon takes ONE focal length. The MLX90640's pixels are not
    # angularly square (55/32 = 1.72 deg across, 35/24 = 1.46 along), so the
    # v coordinate is rescaled into the across-track focal length's units
    # instead of changing that function, which is covered by its own tests and
    # used by the offline path.
    cu, cv = cols / 2.0, rows / 2.0
    out = []
    for i, b in enumerate(blobs):
        u = float(b["col"]) + 0.5
        v = cv + (float(b["row"]) + 0.5 - cv) * (fx / fy)
        try:
            clat, clon = pixel_to_latlon(
                u, v, (cu, cv), fx,
                # NEGATED on purpose. `yaw_deg` here is a COMPASS HEADING --
                # degrees true, clockwise from north -- because that is what a
                # GNSS course and an operator both mean by "heading".
                # pixel_to_latlon's yaw is a mathematical rotation in an
                # (East, North) plane, which runs counter-clockwise, so the two
                # are opposite in sign. Getting this wrong mirrors every
                # contact about the aircraft's north-south axis, which looks
                # entirely plausible on screen and is wrong by up to 180 deg.
                roll=0.0, pitch=0.0, yaw=math.radians(-yaw_deg),
                altitude_m=float(agl_m), lat0=float(lat), lon0=float(lon),
            )
        except ValueError:
            continue

        # Range and bearing relative to the aircraft, in metres. Reported
        # separately from lat/lon because the range survives an unknown
        # heading and the bearing does not.
        east = (clon - lon) * 111_320.0 * math.cos(math.radians(lat))
        north = (clat - lat) * 111_320.0
        rng = math.hypot(east, north)
        brg = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0

        # TWO error numbers, because they answer two different questions and
        # conflating them is how a map ends up drawing the same uncertainty
        # twice.
        #
        #   rel_err_m -- how well the contact is known RELATIVE TO THE
        #     AIRCRAFT. One thermal pixel on the ground (the GSD) plus the
        #     blob's own extent, since a 16 px blob is a body and not a point.
        #     This is what the map should draw around the pin: the aircraft's
        #     own ring already carries the absolute part, and any error in the
        #     aircraft's position moves the drone and the contact together.
        #
        #   err_m -- what the position is worth ON THE GROUND to a team
        #     walking to it. The above, plus the aircraft's own fix.
        #
        # Neither covers an unknown HEADING. That is not an error bar, it is a
        # missing dimension: with no compass the contact could be anywhere on a
        # circle of radius `range_m` about the aircraft. The caller is told via
        # `assumed` and must draw that locus rather than pretend to a bearing.
        gsd = 2.0 * agl_m * math.tan(math.radians(fov_deg[0]) / 2.0) / cols
        fix_err = (max(1.0, hdop or 1.0) * UERE_M) if pos_source == "gps" else 25.0
        blob_r = gsd * math.sqrt(max(1, int(b.get("px", 1))) / math.pi)
        rel_err = math.hypot(gsd, blob_r)
        err = math.hypot(rel_err, fix_err)

        out.append({
            "i": i,
            "lat": clat, "lon": clon,
            "range_m": rng, "bearing_deg": brg,
            "err_m": err, "rel_err_m": rel_err, "gsd_m": gsd,
            "z": b.get("z"), "T": b.get("T"), "px": b.get("px"),
            "assumed": list(assumed),
        })
    return out
