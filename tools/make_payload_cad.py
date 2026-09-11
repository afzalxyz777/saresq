"""SaResQ payload tray, sensor mount and GPS mast as solid CAD (STEP) + STL.

    ./.venv/bin/python tools/make_payload_cad.py --out results/mount/
    ./.venv/bin/python tools/make_payload_cad.py --only mount     # one part

Supersedes make_payload_tray.py and make_sensor_mount.py, which built triangle
meshes. This builds B-rep solids in CadQuery, so it exports **STEP** as well as
STL: a print shop can slice the STL, and anyone can open the STEP in Fusion,
SolidWorks or FreeCAD and edit it. A mesh cannot be meaningfully edited --
change one dimension and you are re-triangulating, not modifying a model.

## Why the sensor mount moved here

It was the last part still coming out of the trimesh script, and it showed red
error faces in every viewer. Measured on the shipped file: **not watertight,
105 broken faces, 30 edges shared by more than two faces, 2 zero-area
triangles**. The cause is structural, not a bad number -- trimesh built each cut
tool with `util.concatenate`, which glues meshes together *without* a boolean
union, so every slot handed the difference operation a self-intersecting solid
with interior faces. The tray and mast, built here, came out watertight with
zero non-manifold edges. An exact solid kernel cannot produce that class of
defect at all, which is why the fix is a rebuild rather than a mesh repair.

The mount now shares `_optical_band_cuts()` with the tray, so the 2 g fit-check
part is cut from *the same* pocket, aperture and slot geometry as the 24 g tray.
If a board drops into the fit check, it drops into the tray. Previously the two
carried independently maintained copies of those numbers and had already drifted.

## Material: PLA is fine for the September submission, not for December

The Kolkata shop stocks PLA, not PETG, so PLA is the default. What that trades:

* **Heat is the only serious loss.** PLA's glass transition is 55-60 C; PETG's is
  ~80 C. A dark part in Kolkata sun passes 55 C easily, and then PLA creeps under
  load rather than failing outright -- the tray sags, the optical baseline shifts,
  and the registration budget goes with it. **Print it WHITE or light grey, not
  black.** Colour is worth more here than anything else on this list, and costs
  nothing. The Pi itself is not the threat: 6 mm standoffs, 7 mm contact pads and
  the vent grilles already keep the plate far below the SoC's 80 C throttle point.
* **Impact is the other loss.** PLA is brittle where PETG is tough, so a hard
  landing cracks it instead of bending it. Print 4 perimeters instead of 3.
* **Stiffness is a GAIN.** PLA's modulus is ~3.5 GPa against PETG's ~2.1, so the
  plate flexes less between the sensor band and the Pi -- which is what the
  optical baseline actually cares about.
* **Print quality is a GAIN**, and it matters most on the part that matters most.
  PETG strings and rounds off corners; PLA holds dimensions better, so the board
  pockets come out closer to nominal. On a 145 x 135 flat plate PLA also warps
  least -- ABS would lift at the corners without an enclosure, which is why ABS is
  NOT the fallback here even though shops stock it. If a shop offers a real
  alternative, ASA or PETG-CF, take it; plain ABS is a step sideways.

So: **PLA for the fit check and for the SIH submission** (deck and video, no
flight). Source PETG or ASA before the December Grand Finale, when the payload
actually flies and sits in the sun between sorties. The geometry does not change
-- only the filament -- so this is a reprint, not a redesign.

One PLA-specific slicer setting: the board pockets open onto the BED, so
elephant's foot squeezes the pocket walls inward. Turn on elephant-foot
compensation at 0.15-0.2 mm, or the 0.4 mm clearance per side gets eaten. This is
exactly what the 2.6 g fit-check part exists to catch.

## Two geometry rules this file now enforces rather than assumes

* **Corner bolts sit on the fillet arc centre.** See `_corner_bolts`.
* **Blind pockets overshoot the open face by a real amount**, not by 0.001 mm.
  See `_pocket`.

Both are checked at build time, and every exported STL is re-loaded and verified
watertight, manifold and volume-consistent before the script will exit 0.

Every dimension below traces to a real part or a computed requirement; nothing
is styling. The reasoning for the big ones:

* **28 mm optical baseline** -- parallax is baseline/range, and an MLX90640 pixel
  subtends 30 mrad (55 deg / 32 px). 28 mm gives 0.047 px of parallax at the 20 m
  survey altitude and 0.467 px at a 2 m bench test, both inside the 0.5 px
  registration budget in configs/pipeline.yaml.
* **Flat, not stacked** -- the 209 g power bank dominates a 335 g payload. Stacked,
  that mass sits high and the payload pendulums. It also buries the Pi.
* **Thermal break behind the sensor band** -- the MLX90640 reports scene
  temperature relative to its own die, compensated by an on-chip Ta sensor. Let a
  5 W Pi warm that die and the compensation lags: every reading skews, and it
  skews plausibly rather than failing.
* **Vent grille under the Pi** -- a Pi 4 soft-throttles at 80 C. Sustained TFLite
  inference in Kolkata ambient on a board sitting over a solid plate gets there.
* **microSD notches at both short ends of the Pi** -- the card slot is on the
  underside and the board sits 6 mm up. Without a notch, re-flashing means
  unbolting the Pi.
* **Elongated screw slots everywhere a breakout mounts** -- clone boards put their
  holes in slightly different places, and a round hole 1.5 mm out is a board that
  will not bolt down.
"""
from __future__ import annotations

import argparse
import math
import pathlib

import cadquery as cq

# --- Components (mm) ---------------------------------------------------------
PI_W, PI_L = 56.0, 85.0
# 58 x 49 is the Pi 4 pattern, but 58 mm runs along the 85 mm LENGTH and 49 mm
# across the 56 mm width. In this layout PI_W is X and PI_L is Y, so they map
# DX=49, DY=58. Swapped, the standoffs sit 58 mm apart across a 56 mm board --
# hanging off both plate edges and matching no hole on the Pi.
PI_HOLE_DX, PI_HOLE_DY = 49.0, 58.0
BANK_W, BANK_L, BANK_H = 80.0, 90.0, 28.0
THERMAL_W, THERMAL_H = 25.0, 22.0
CAMERA_W, CAMERA_H = 25.0, 24.0
IMU_W, IMU_H = 21.5, 16.5
GPS_W, GPS_H = 23.0, 33.0
GPS_ANT = 25.0

PI_MASS_G, BANK_MASS_G = 46.0, 209.0
IMU_MASS_G, GPS_MASS_G, GPS_ANT_MASS_G = 5.0, 9.0, 14.0

# --- Tray --------------------------------------------------------------------
PLATE_Z, MARGIN, SENSOR_BAND, GAP = 3.0, 3.0, 36.0, 3.0
PLATE_X = MARGIN + PI_W + GAP + BANK_W + MARGIN            # 145
PLATE_Y = MARGIN + SENSOR_BAND + GAP + max(PI_L, BANK_L) + MARGIN  # 135
# Corner radius is not styling: it sets how much material surrounds the corner
# bolts, which is what holds the payload to the airframe. See _corner_bolts.
CORNER_R = 6.0
MIN_CORNER_WALL = 2.5

CLR, POCKET_D = 0.4, 1.8
BASELINE = 28.0
THERMAL_AP, CAMERA_AP = 11.0, 9.0
STANDOFF_D, STANDOFF_H, PI_TAP = 7.0, 6.0, 2.6
WALL_H, WALL_T = 6.0, 2.5
ZIP_W, ZIP_H = 4.5, 2.6
M3 = 3.4
VENT_W, VENT_PITCH = 4.0, 8.5
SD_W, SD_L = 26.0, 13.0

# The vent slots run front-to-back under the Pi and the microSD notches cut
# across both of its short ends, so the two features race each other for the
# same strip of plate. At VENT_L = 62 they overlapped by 0.5 mm at each end --
# just enough to sever the strips lying wholly inside the notch width, leaving
# TWO LOOSE 4.5 x 61 x 3 mm slivers that a slicer prints as separate floating
# sticks. So the vent length is derived from the notch position, never typed in.
SD_INNER = PI_L / 2 - SD_L + 1.0        # notch inner edge, from the Pi centre
VENT_BRIDGE = 2.5                       # material left spanning each end
VENT_L = 2 * (SD_INNER - VENT_BRIDGE)   # -> 56.0
SCREW_D, SLOT_LEN = 2.4, 3.0
TIE_W, TIE_H = 3.2, 2.2
LIGHTEN_D = 12.0

# A blind-pocket cutter overshoots the face it opens through by this much.
# It used to be 0.001 mm, which is below the tolerance OCC fuses coincident
# faces at AND below float32 STL resolution at these coordinates -- so instead
# of a clean opening it left a sliver. 1 mm cannot be mistaken for coincidence.
POCKET_OVER = 1.0

# --- Sensor mount (the fit-check part) ---------------------------------------
# Same PLATE_Z, pockets, apertures and slots as the tray's sensor band, so this
# validates the tray rather than merely resembling it.
#
# 72 x 36, not the 66 x 34 this part used to be. At 66 x 34 all four corner M3s
# cut INTO a sensor pocket -- 0.45 mm into the thermal pocket, 0.60 mm into the
# camera pocket. The board would then sit over a notch and the bolt would have
# no shoulder on its inboard side. The mesh was clean, so nothing flagged it;
# only the clearance arithmetic in _audit_clearances does.
MOUNT_X, MOUNT_Y = 72.0, 36.0
MOUNT_CORNER_R = 5.0
CAM_RELIEF_W, THERM_RELIEF_W = 18.0, 14.0

# Thinnest wall allowed between two features: 3 perimeters at a 0.4 mm nozzle.
# Below this the slicer drops to a gap-fill bead, which is weak and inconsistent.
MIN_FEATURE_WALL = 1.2

# STL tessellation. CadQuery's default linear deflection is 0.1 mm, which visibly
# facets an 9 mm lens aperture; 0.02 mm is a quarter of one 0.08 mm layer and
# well under any FDM machine's positioning resolution, so the sliced result is
# limited by the printer rather than by the mesh.
STL_TOL, STL_ANG = 0.02, 0.1

# Filament density (g/cm3). PRINT_FILL is the fraction of the solid volume that
# actually gets extruded at 25% infill with 3 perimeters -- roughly 45%.
DENSITY = {"petg": 1.27, "pla": 1.24, "asa": 1.07, "abs": 1.04}
PRINT_FILL = 0.45

# Layout anchors
Y_FRONT = -PLATE_Y / 2 + MARGIN + SENSOR_BAND / 2
Y_REAR = Y_FRONT + SENSOR_BAND / 2 + GAP
X_PI = -PLATE_X / 2 + MARGIN + PI_W / 2
X_BANK = PLATE_X / 2 - MARGIN - BANK_W / 2
Y_PI = Y_REAR + PI_L / 2
Y_BANK = Y_REAR + BANK_L / 2
Y_BREAK = Y_FRONT + SENSOR_BAND / 2 - 1.0


def _box(w, l, h, x, y, z):
    return cq.Workplane("XY").box(w, l, h).translate((x, y, z))


def _cyl(d, h, x, y, z):
    return cq.Workplane("XY").cylinder(h, d / 2).translate((x, y, z))


def _slot(x, y, length=SLOT_LEN, d=SCREW_D):
    """Capsule slot through the plate, elongated in X."""
    s = _box(length, d, PLATE_Z * 4, x, y, 0)
    return s.union(_cyl(d, PLATE_Z * 4, x - length / 2, y, 0)) \
            .union(_cyl(d, PLATE_Z * 4, x + length / 2, y, 0))


def _fuse(solids):
    out = solids[0]
    for s in solids[1:]:
        out = out.union(s)
    return out


def _pocket(w, l, depth, x, y, face):
    """Blind pocket cut into the top (face=+1) or bottom (face=-1) face.

    The cutter overshoots the open face by POCKET_OVER and stops exactly at the
    pocket floor, so the floor depth is set by arithmetic and the opening is set
    by a cut that is unambiguously through. Doing it with a 0.001 mm epsilon (as
    this did) leaves geometry that is neither coincident nor cleanly separated.
    """
    h = depth + POCKET_OVER
    return _box(w, l, h, x, y, face * (PLATE_Z / 2 - (depth - POCKET_OVER) / 2))


def _corner_bolts(plate_x, plate_y, corner_r, d=M3):
    """Airframe M3s, one per corner, centred ON each fillet's arc centre.

    Near a corner the nearest material boundary is the fillet arc, not either
    straight edge -- so an inset-from-the-edges placement puts the hole on the
    diagonal where there is least material. At inset i the hole reaches
    sqrt(2)*|corner_r - i| + d/2 from the arc centre, and that has to stay under
    corner_r. It did not:

        part          corner_r   inset   reach    wall
        sensor mount    3.0       4.5    3.82    -0.82 mm   <- open notch
        tray            4.0       6.0    4.53    -0.53 mm   <- open notch

    Both bolts broke out through the corner and printed as C-shaped notches, and
    these are the four screws carrying the payload on the airframe. The mount's
    mesh was so damaged that this never showed; the tray's was a valid solid, so
    it never showed either. Neither was visible without doing the arithmetic.

    Putting the hole exactly on the arc centre sets the distance term to zero,
    which is both the maximum achievable wall and a uniform one -- the material
    is then corner_r - d/2 thick in every direction around the corner, against
    the straight edges included. That makes the wall a function of the corner
    radius alone, which the assertion below can then enforce.
    """
    wall = corner_r - d / 2
    assert wall >= MIN_CORNER_WALL, (
        f"corner radius {corner_r} leaves only {wall:.2f} mm around an M3 "
        f"(need {MIN_CORNER_WALL}); raise the corner radius, do not inset the hole")
    return [_cyl(d, PLATE_Z * 4, sx * (plate_x / 2 - corner_r), sy * (plate_y / 2 - corner_r), 0)
            for sx in (-1, 1) for sy in (-1, 1)]


def _audit_clearances(part, holes, rects):
    """Assert every round hole keeps a printable wall to every board pocket.

    Solid modellers merge overlapping features silently and produce a perfectly
    valid, perfectly wrong solid -- which is exactly how the GPS pocket once
    ended up sitting over a thermal-break slot, and how all four of the sensor
    mount's corner bolts ended up cutting into its sensor pockets. Neither shows
    up in a mesh check, in `isValid()`, or in a render at any angle that happens
    to be chosen. Only the arithmetic sees it, so the arithmetic runs every build.

    holes: (label, x, y, diameter). rects: (label, x, y, width, length).
    """
    worst = (float("inf"), "")
    for hlabel, hx, hy, hd in holes:
        for rlabel, cx, cy, w, l in rects:
            # Distance from the hole centre to the rectangle (0 if inside).
            gap = math.hypot(max(abs(hx - cx) - w / 2, 0.0),
                             max(abs(hy - cy) - l / 2, 0.0)) - hd / 2
            if gap < worst[0]:
                worst = (gap, f"{hlabel} to {rlabel}")
            if gap < MIN_FEATURE_WALL:
                raise SystemExit(
                    f"{part}: {hlabel} leaves {gap:+.2f} mm to {rlabel} "
                    f"(need {MIN_FEATURE_WALL}). "
                    + ("They intersect -- the pocket floor would be notched and the "
                       "bolt would lose its shoulder." if gap < 0 else
                       "Too thin to print as a wall."))
    return worst


def _optical_band_cuts(y):
    """Thermal + camera pockets, lens apertures and board slots, about y.

    Shared verbatim by the tray and the sensor mount. These are the only
    dimensions where a fit check is meaningful, so they must come from one
    place: a fit-check part cut from its own copy of the numbers checks the fit
    of the fit-check part.
    """
    tw, th = THERMAL_W + 2 * CLR, THERMAL_H + 2 * CLR
    cw, ch = CAMERA_W + 2 * CLR, CAMERA_H + 2 * CLR

    # Pockets open DOWNWARD so both lenses look at the ground; the apertures
    # then go straight through and nothing bridges over open air when printed.
    cuts = [_pocket(tw, th, POCKET_D, -BASELINE / 2, y, -1),
            _pocket(cw, ch, POCKET_D, BASELINE / 2, y, -1),
            _cyl(THERMAL_AP, PLATE_Z * 4, -BASELINE / 2, y, 0),
            _cyl(CAMERA_AP, PLATE_Z * 4, BASELINE / 2, y, 0)]
    for px, ph in ((-BASELINE / 2, th), (BASELINE / 2, ch)):
        for sy in (-1, 1):
            cuts.append(_slot(px, y + sy * (ph / 2 - 2.8)))
    return cuts


def build_sensor_mount():
    """Two-sensor plate: print this ~2 g part FIRST as a fit check.

    It is the tray's sensor band, cut to a 66 x 34 plate. Ten minutes and a few
    rupees of filament confirm that the MLX90640 and the Pi Camera actually drop
    into the pockets and that the M2 slots line up with their real holes, before
    committing to the 24 g, multi-hour tray print.
    """
    tw, th = THERMAL_W + 2 * CLR, THERMAL_H + 2 * CLR
    cw, ch = CAMERA_W + 2 * CLR, CAMERA_H + 2 * CLR

    _audit_clearances(
        "sensor mount",
        [(f"corner M3 ({sx:+.0f},{sy:+.0f})",
          sx * (MOUNT_X / 2 - MOUNT_CORNER_R), sy * (MOUNT_Y / 2 - MOUNT_CORNER_R), M3)
         for sx in (-1, 1) for sy in (-1, 1)],
        [("thermal pocket", -BASELINE / 2, 0.0, tw, th),
         ("camera pocket", BASELINE / 2, 0.0, cw, ch)])

    plate = (cq.Workplane("XY").box(MOUNT_X, MOUNT_Y, PLATE_Z)
             .edges("|Z").fillet(MOUNT_CORNER_R))

    cuts = _optical_band_cuts(0.0)
    cuts += _corner_bolts(MOUNT_X, MOUNT_Y, MOUNT_CORNER_R)

    # Cable reliefs, opening the rear edge into each pocket: the camera's FFC and
    # the thermal board's jumpers exit flat instead of being folded back over a
    # rib. Each starts 2 mm inside its pocket wall so both M2 slots survive, and
    # is narrower than its pocket so the board is still captured sideways.
    for px, relief_w, ph in ((BASELINE / 2, CAM_RELIEF_W, ch),
                             (-BASELINE / 2, THERM_RELIEF_W, th)):
        y0 = ph / 2 - 2.0
        y1 = MOUNT_Y / 2 + POCKET_OVER
        cuts.append(_box(relief_w, y1 - y0, PLATE_Z * 4, px, (y0 + y1) / 2, 0))

    return plate.cut(_fuse(cuts))


def build_tray():
    tray = cq.Workplane("XY").box(PLATE_X, PLATE_Y, PLATE_Z).edges("|Z").fillet(CORNER_R)

    # Pi standoffs, added before any cutting.
    for sx in (-1, 1):
        for sy in (-1, 1):
            tray = tray.union(_cyl(STANDOFF_D, STANDOFF_H,
                                   X_PI + sx * PI_HOLE_DX / 2, Y_PI + sy * PI_HOLE_DY / 2,
                                   PLATE_Z / 2 + STANDOFF_H / 2))
    # Power-bank retaining walls, open on the side facing the Pi so the
    # inbuilt USB-C lead can reach and the bank slides out to charge.
    for sx in (-1, 1):
        tray = tray.union(_box(WALL_T, BANK_L, WALL_H,
                               X_BANK + sx * (BANK_W / 2 + WALL_T / 2), Y_BANK,
                               PLATE_Z / 2 + WALL_H / 2))
    tray = tray.union(_box(BANK_W + 2 * WALL_T, WALL_T, WALL_H,
                           X_BANK, Y_BANK + BANK_L / 2 + WALL_T / 2, PLATE_Z / 2 + WALL_H / 2))

    iw, ih = IMU_W + 2 * CLR, IMU_H + 2 * CLR
    gw, gh = GPS_H + 2 * CLR, GPS_W + 2 * CLR        # rotated 90 deg: see below

    # Thermal + camera pockets, apertures and slots -- the geometry the sensor
    # mount fit-checks, so it is generated once and used by both parts.
    cuts = _optical_band_cuts(Y_FRONT)

    # IMU and GPS: pockets open UPWARD -- neither looks at the ground. The IMU
    # sits beside the optics on purpose: its attitude solution projects pixels
    # to ground coordinates, so a short lever arm to the optical axes means a
    # smaller error between what the camera saw and what the IMU reported.
    #
    # The GPS module is rotated 90 deg. Portrait it is 33.8 mm deep in a 36 mm
    # band whose rear 7 mm is the thermal break, so a break slot would cut
    # through its pocket floor and the module would sit over a hole.
    cuts += [_pocket(iw, ih, POCKET_D, -52.0, Y_FRONT, +1),
             _pocket(gw, gh, POCKET_D, 52.0, Y_FRONT, +1),
             _box(8.0, 4.0, PLATE_Z * 4, -52.0, Y_FRONT + ih / 2 + 1.0, 0),
             _box(8.0, 4.0, PLATE_Z * 4, 52.0, Y_FRONT + gh / 2 + 1.0, 0)]

    # Elongated screw slots for the IMU and GPS (the optics' slots come from
    # _optical_band_cuts above).
    for px, ph in ((-52.0, ih), (52.0, gh)):
        for sy in (-1, 1):
            cuts.append(_slot(px, Y_FRONT + sy * (ph / 2 - 2.8)))

    # Thermal break: three ribs, four gaps. Doubles as the camera-ribbon and
    # jumper-wire pass-through, and as extra vent area.
    for sx in (-1.5, -0.5, 0.5, 1.5):
        cuts.append(_box(26.0, 7.0, PLATE_Z * 4, sx * 30.0, Y_BREAK, 0))

    # Pi tapping holes.
    for sx in (-1, 1):
        for sy in (-1, 1):
            cuts.append(_cyl(PI_TAP, STANDOFF_H * 4,
                             X_PI + sx * PI_HOLE_DX / 2, Y_PI + sy * PI_HOLE_DY / 2, PLATE_Z / 2))

    # Vent grilles. Slots run front-to-back so the tray vents under downward
    # rotor wash AND in forward flight.
    for i in range(-4, 5):
        cuts.append(_box(VENT_W, VENT_L, PLATE_Z * 4, X_PI + i * VENT_PITCH, Y_PI, 0))
    for i in range(-3, 4):
        cuts.append(_box(VENT_W, BANK_L * 0.62, PLATE_Z * 4,
                         X_BANK + i * (VENT_PITCH + 2.0), Y_BANK, 0))

    # microSD finger access, both short ends of the Pi.
    for sy in (-1, 1):
        cuts.append(_box(SD_W, SD_L, PLATE_Z * 4, X_PI,
                         Y_PI + sy * (PI_L / 2 - SD_L / 2 + 1.0), 0))

    # Power-bank zip ties.
    for sy in (-0.28, 0.28):
        for sx in (-1, 1):
            cuts.append(_box(ZIP_W, ZIP_H, PLATE_Z * 4,
                             X_BANK + sx * (BANK_W / 2 - 4.0), Y_BANK + sy * BANK_L, 0))

    # Cable tie-downs along the sensor->GPIO route and beside the bank.
    for ty in (Y_FRONT + 22.0, Y_PI - 18.0, Y_PI + 20.0):
        for sx in (-1, 1):
            cuts.append(_box(TIE_W, TIE_H, PLATE_Z * 4, X_PI + sx * 33.0, ty, 0))
    for ty in (Y_BANK - 26.0, Y_BANK + 26.0):
        cuts.append(_box(TIE_W, TIE_H, PLATE_Z * 4, X_BANK - BANK_W / 2 - 5.5, ty, 0))

    # Lightening holes in the dead strip between Pi and bank.
    x_gap = (X_PI + PI_W / 2 + X_BANK - BANK_W / 2) / 2
    for i in range(-2, 3):
        cuts.append(_cyl(LIGHTEN_D, PLATE_Z * 4, x_gap, Y_BANK + i * 17.0, 0))

    # Airframe interface. Same audit as the mount: the tray's corners are far
    # from the optics but close to the GPS pocket at x = +52.
    _audit_clearances(
        "tray",
        [(f"corner M3 ({sx:+.0f},{sy:+.0f})",
          sx * (PLATE_X / 2 - CORNER_R), sy * (PLATE_Y / 2 - CORNER_R), M3)
         for sx in (-1, 1) for sy in (-1, 1)],
        [("thermal pocket", -BASELINE / 2, Y_FRONT, THERMAL_W + 2 * CLR, THERMAL_H + 2 * CLR),
         ("camera pocket", BASELINE / 2, Y_FRONT, CAMERA_W + 2 * CLR, CAMERA_H + 2 * CLR),
         ("IMU pocket", -52.0, Y_FRONT, iw, ih),
         ("GPS pocket", 52.0, Y_FRONT, gw, gh)])
    cuts += _corner_bolts(PLATE_X, PLATE_Y, CORNER_R)
    for sy in (-1, 1):
        cuts.append(_box(ZIP_W, ZIP_H, PLATE_Z * 4, 0, sy * (PLATE_Y / 2 - CORNER_R), 0))

    return tray.cut(_fuse(cuts))


def build_mast():
    """Raised pad for the NEO-6M ceramic patch antenna -- a separate part.

    The antenna is the one component that must not live in the payload. It needs
    sky view while everything else looks down, and a Pi 4 radiates broadband RF
    right across the GPS L1 band at 1575 MHz -- the HDMI serialiser especially.
    A patch sitting beside it loses satellites, and the symptom is a slow or
    wandering fix rather than an obvious failure. Standard practice is a mast.

    Constant C-section extruded vertically: every layer is identical, so it
    prints with no supports in any orientation.
    """
    foot, riser, pad, t, w = 30.0, 45.0, 32.0, 3.0, 32.0
    body = (_box(foot, w, t, foot / 2, 0, t / 2)
            .union(_box(t, w, riser, t / 2, 0, t + riser / 2))
            .union(_box(pad, w, t, pad / 2, 0, t + riser + t / 2)))
    for sx in (0.30, 0.72):
        body = body.cut(_cyl(M3, t * 4, foot * sx, 0, t / 2))
    for sy in (-1, 1):
        body = body.cut(_box(ZIP_W, ZIP_H, t * 4, pad * 0.62, sy * (w / 2 - 4.0),
                             t + riser + t / 2))
    return body


def verify_stl(path, solid_volume_mm3):
    """Re-load an exported STL and prove it is printable. Returns a list of faults.

    This is the check that was missing. The old mount was exported and shipped
    without anything ever reading it back, so 105 broken faces reached the STL,
    the viewer and nearly the print shop. Every fault below is one a slicer draws
    in red, and every one is cheap to detect:

    * boundary edges  -- an edge used by exactly one triangle is a hole in the
      surface, so the solid has no defined inside. Slicers guess, and guess
      differently at each layer.
    * non-manifold    -- an edge used by more than two triangles. Two surfaces
      meet along it and neither is the boundary.
    * degenerate      -- zero-area triangles carry no normal.
    * inconsistent winding / negative volume -- inside-out normals.
    * disconnected pieces -- the part must be ONE connected solid. Overlapping
      cuts can isolate an island of material, and nothing else here notices: the
      result is still watertight, still manifold, still a valid solid, and still
      the right volume. It simply falls off the bed as a loose stick. The tray
      shipped two such slivers where the vent slots met the microSD notches.
    * volume drift    -- the surest whole-part check. The bound is two-sided,
      not one-sided: a tessellated arc is a polygon inscribed in the true arc,
      which removes material on a convex boundary (the rounded corners) but ADDS
      it on a concave one (every hole and aperture, of which this part is mostly
      made). So the sign carries no information; only the magnitude does, and at
      STL_TOL = 0.02 mm it lands near 0.001%. A drift of any real size means the
      mesh is not this solid.
    """
    import numpy as np
    import trimesh

    m = trimesh.load(str(path))
    _, counts = np.unique(m.edges_sorted, axis=0, return_counts=True)
    tri = m.triangles
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2.0
    drift = (m.volume - solid_volume_mm3) / solid_volume_mm3

    faults = []
    if int((counts == 1).sum()):
        faults.append(f"{int((counts == 1).sum())} boundary edges (not watertight)")
    if int((counts > 2).sum()):
        faults.append(f"{int((counts > 2).sum())} non-manifold edges")
    if int((area < 1e-9).sum()):
        faults.append(f"{int((area < 1e-9).sum())} degenerate triangles")
    if not m.is_winding_consistent:
        faults.append("inconsistent winding")
    pieces = m.split(only_watertight=False)
    if len(pieces) > 1:
        loose = sorted(pieces, key=lambda p: -p.volume)[1:]
        detail = ", ".join(f"{p.volume / 1000:.2f} cm3 at "
                           f"x {p.bounds[0][0]:+.0f}..{p.bounds[1][0]:+.0f} "
                           f"y {p.bounds[0][1]:+.0f}..{p.bounds[1][1]:+.0f}"
                           for p in loose[:4])
        faults.append(f"{len(pieces)} disconnected pieces -- loose: {detail}")
    if m.volume <= 0:
        faults.append("negative volume (normals inverted)")
    if abs(drift) > 1e-3:
        faults.append(f"volume drift {drift * 100:+.3f}% vs the B-rep solid")
    return faults, len(m.faces), drift


PARTS = {
    "saresq_payload_tray": build_tray,
    "saresq_sensor_mount": build_sensor_mount,
    "saresq_gps_mast": build_mast,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mount")
    ap.add_argument("--material", default="pla", choices=sorted(DENSITY),
                    help="affects the printed-mass estimate only, not the geometry. "
                         "PLA is the default because that is what the Kolkata shop "
                         "stocks; see the module docstring for what that costs.")
    ap.add_argument("--only", default="",
                    help="comma-separated subset: tray, mount, mast. Building one "
                         "part is cheap; the tray boolean is not, and this machine "
                         "is memory-bound while a training run is live.")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wanted = [w.strip() for w in args.only.split(",") if w.strip()]
    names = [n for n in PARTS if not wanted or any(w in n for w in wanted)]
    if not names:
        raise SystemExit(f"--only {args.only!r} matched no part of {sorted(PARTS)}")

    t_ifov = math.radians(55.0) / 32
    total_tray_g = 0.0
    all_faults = {}

    for name in names:
        solid = PARTS[name]()
        shape = solid.val()
        if not shape.isValid():
            raise SystemExit(f"{name}: CadQuery produced an invalid solid; "
                             f"do not export it")
        cq.exporters.export(solid, str(out / f"{name}.step"))
        cq.exporters.export(solid, str(out / f"{name}.stl"),
                            tolerance=STL_TOL, angularTolerance=STL_ANG)

        vol_mm3 = shape.Volume()
        vol = vol_mm3 / 1000.0
        grams = vol * DENSITY[args.material] * PRINT_FILL
        # The sensor mount is a bench fit check, not a flight part -- the tray
        # carries those sensors in the air. Counting it would inflate the mass
        # budget by a part that never leaves the desk.
        if name != "saresq_sensor_mount":
            total_tray_g += grams
        bb = shape.BoundingBox()
        faults, tris, drift = verify_stl(out / f"{name}.stl", vol_mm3)
        all_faults[name] = faults

        print(f"  {name}")
        print(f"    envelope   {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm")
        print(f"    volume     {vol:.2f} cm3   -> {grams:.1f} g "
              f"{args.material.upper()} @ 25% infill")
        print(f"    mesh       {tris:,} triangles, {drift * 100:+.4f}% vs solid")
        verdict = ("CLEAN -- watertight, manifold, no degenerate faces"
                   if not faults else "FAULTS: " + "; ".join(faults))
        print(f"    check      {verdict}")

    print(f"\n  optical baseline      {BASELINE:.0f} mm")
    print(f"  parallax @ 20 m       {(BASELINE/1000/20)/t_ifov:.3f} thermal px")
    print(f"  parallax @  2 m       {(BASELINE/1000/2)/t_ifov:.3f} thermal px  (budget 0.5)")
    for r, part in ((MOUNT_CORNER_R, "sensor mount"), (CORNER_R, "tray")):
        print(f"  corner wall, {part:<12s} {r - M3 / 2:.1f} mm around each M3")
    if {"saresq_payload_tray", "saresq_gps_mast"} <= set(names):
        hw = PI_MASS_G + BANK_MASS_G + IMU_MASS_G + GPS_MASS_G + GPS_ANT_MASS_G + 8 + 20
        print(f"  printed parts         {total_tray_g:.1f} g")
        print(f"  hardware + cables     {hw:.1f} g")
        print(f"  TOTAL PAYLOAD         {total_tray_g + hw:.1f} g")

    # Exit non-zero on any fault. A defective STL that still lands in
    # results/mount/ is how the broken mount reached the viewer in the first
    # place, so the failure has to be loud and the exit code has to carry it.
    broken = {n: f for n, f in all_faults.items() if f}
    if broken:
        raise SystemExit("\nNOT PRINTABLE -- do not send these to the shop:\n"
                         + "\n".join(f"  {n}: {'; '.join(f)}" for n, f in broken.items()))
    print(f"\n  all {len(names)} exported STL(s) verified printable.")


if __name__ == "__main__":
    main()
