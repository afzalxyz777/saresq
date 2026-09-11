"""Generate the full SaResQ payload tray as a printable STL.

    ./.venv/bin/python tools/make_payload_tray.py --out results/mount/

Carries the whole perception payload on one plate: MLX90640 + Pi Camera v2
looking down through the front, Raspberry Pi 4 and the Callmate CT-110 power
bank behind them.

## Why flat, not stacked

"Most compact" pulls two ways, and for a flown payload the flat answer wins:

* **CG.** The power bank is 209 g against the Pi's 46 g -- it dominates. Stacked,
  that mass sits high and the payload pendulums under the airframe. Flat, it sits
  in the same plane as everything else.
* **Airflow.** The 1 GB Pi 4 runs TFLite inference, camera capture and I2C polling
  concurrently. Burying it under a 209 g battery is how you get thermal throttling
  mid-survey, and `vcgencmd get_throttled` has read 0x0 so far only because the
  board has been idle and in open air.
* **Printability.** A flat tray needs zero supports and prints in one piece on any
  hobby bed. A stack needs either supports through the sensor apertures or a
  multi-part assembly with fasteners.

## Everything is retained, nothing is enclosed

The power bank sits in a walled pocket held by zip ties, not a captive box: it has
to come out to recharge. The Pi bolts to standoffs so its ports and microSD stay
reachable. Sensor boards drop into edge-located pockets -- see make_sensor_mount.py
for why edges, not screw holes.
"""
from __future__ import annotations

import argparse
import math
import pathlib

import trimesh

# --- Component dimensions (mm), from the real parts --------------------------
PI_W, PI_L = 56.0, 85.0                 # Raspberry Pi 4 Model B PCB
PI_HOLE_DX, PI_HOLE_DY = 58.0, 49.0     # M2.5 mounting pattern
PI_HOLE_INSET = 3.5

BANK_W, BANK_L, BANK_H = 80.0, 90.0, 28.0   # Callmate CT-110, 209 g
BANK_MASS_G = 209.0
PI_MASS_G = 46.0

THERMAL_W, THERMAL_H = 25.0, 22.0       # 7SEMI MLX9064X (measured)
CAMERA_W, CAMERA_H = 25.0, 24.0         # Pi Camera v2 (25.0 x 23.86 published)
IMU_W, IMU_H = 21.5, 16.5               # GY-521 MPU-6050 breakout
GPS_W, GPS_H = 23.0, 33.0               # GY-NEO6MV2 module board
GPS_ANT = 25.0                          # ceramic patch antenna, on a U.FL lead
IMU_MASS_G, GPS_MASS_G, GPS_ANT_MASS_G = 5.0, 9.0, 14.0
BASELINE = 28.0                         # optical centre-to-centre; see make_sensor_mount.py

# --- Tray ---------------------------------------------------------------------
PLATE_Z = 3.0
MARGIN = 3.0
SENSOR_BAND = 36.0                      # front strip depth
GAP = 3.0

PLATE_X = MARGIN + PI_W + GAP + BANK_W + MARGIN          # 145
PLATE_Y = MARGIN + SENSOR_BAND + GAP + max(PI_L, BANK_L) + MARGIN

PCB_CLEARANCE = 0.4
POCKET_DEPTH = 1.8
THERMAL_APERTURE_D = 11.0
CAMERA_APERTURE_D = 9.0

STANDOFF_D, STANDOFF_H = 7.0, 6.0       # lifts the Pi clear for airflow
PI_HOLE_D = 2.6                         # M2.5 tapping
WALL_H, WALL_T = 6.0, 2.5               # power-bank retaining walls
ZIPTIE_W, ZIPTIE_H = 4.5, 2.6
MOUNT_HOLE_D = 3.4                      # M3, airframe
LIGHTEN_D = 12.0
# Ventilation: 4 mm slots on an 8.5 mm pitch. The Pi 4 soft-throttles at 80 C
# and hard-throttles at 85 C; ambient in Kolkata plus sustained inference gets
# there fast on a bare board in still air.
VENT_W, VENT_L, VENT_PITCH = 4.0, 62.0, 8.5
SD_NOTCH_W, SD_NOTCH_L = 26.0, 13.0     # microSD finger access under the Pi
BOARD_SCREW_D = 2.4                     # M2 clearance for the sensor boards
BOARD_SLOT_LEN = 3.0                    # elongated: breakout hole positions vary
TIE_W, TIE_H = 3.2, 2.2                 # cable tie-down slots

PETG_DENSITY = 1.27
PRINT_FILL = 0.45                       # 25% infill + 3 perimeters, effective


def box_at(w, l, h, cx, cy, cz):
    b = trimesh.creation.box(extents=(w, l, h))
    b.apply_translation([cx, cy, cz])
    return b


def cyl_at(d, h, cx, cy, cz):
    c = trimesh.creation.cylinder(radius=d / 2, height=h, sections=40)
    c.apply_translation([cx, cy, cz])
    return c


def build():
    plate = box_at(PLATE_X, PLATE_Y, PLATE_Z, 0, 0, 0)

    y_front = -PLATE_Y / 2 + MARGIN + SENSOR_BAND / 2      # sensor band centre
    y_rear = y_front + SENSOR_BAND / 2 + GAP               # rear block starts here
    x_pi = -PLATE_X / 2 + MARGIN + PI_W / 2
    x_bank = PLATE_X / 2 - MARGIN - BANK_W / 2
    y_pi = y_rear + PI_L / 2
    y_bank = y_rear + BANK_L / 2

    adds, cuts = [], []

    # --- Sensors: pockets opening downward, so the lenses look at the ground.
    tw, th = THERMAL_W + 2 * PCB_CLEARANCE, THERMAL_H + 2 * PCB_CLEARANCE
    cw, ch = CAMERA_W + 2 * PCB_CLEARANCE, CAMERA_H + 2 * PCB_CLEARANCE
    pocket_z = -PLATE_Z / 2 + POCKET_DEPTH / 2 - 0.001
    cuts.append(box_at(tw, th, POCKET_DEPTH, -BASELINE / 2, y_front, pocket_z))
    cuts.append(box_at(cw, ch, POCKET_DEPTH, BASELINE / 2, y_front, pocket_z))
    cuts.append(cyl_at(THERMAL_APERTURE_D, PLATE_Z * 4, -BASELINE / 2, y_front, 0))
    cuts.append(cyl_at(CAMERA_APERTURE_D, PLATE_Z * 4, BASELINE / 2, y_front, 0))
    # Wiring pass-through AND thermal break between the sensor band and the Pi.
    #
    # The MLX90640 is a thermopile: it reports scene temperature relative to its
    # own die, compensated by an on-chip Ta sensor. If its die drifts because a
    # 5 W Pi is conducting heat into the same plate, the compensation lags and
    # every reading skews -- a failure that produces plausible numbers, not an
    # error. PETG conducts poorly (0.2 W/m.K) but the path is still worth
    # cutting, so the plate is reduced to three narrow ribs here. The gaps
    # double as the wiring pass-through and as extra vent area.
    y_break = y_front + SENSOR_BAND / 2 - 1.0
    for sx in (-1.5, -0.5, 0.5, 1.5):
        cuts.append(box_at(26.0, 7.0, PLATE_Z * 4, sx * 30.0, y_break, 0))

    # --- IMU and GPS module: pockets opening UPWARD.
    # Neither needs to see the ground, so unlike the two cameras these sit on
    # the top face. The IMU goes beside the cameras deliberately: its attitude
    # solution is used to project pixels to ground coordinates, so the closer
    # it is to the optical axes, the smaller the lever-arm error between what
    # the camera saw and what the IMU says the payload was doing.
    iw, ih = IMU_W + 2 * PCB_CLEARANCE, IMU_H + 2 * PCB_CLEARANCE
    # GPS module mounted with its LONG axis across the tray. Portrait would be
    # 33.8 mm deep in a 36 mm band whose rear 7 mm is the thermal break, so the
    # pocket floor would be cut through by a break slot -- the module would sit
    # over a hole. Rotating it is free and leaves 1.6 mm of clearance.
    gw, gh = GPS_H + 2 * PCB_CLEARANCE, GPS_W + 2 * PCB_CLEARANCE
    up_z = PLATE_Z / 2 - POCKET_DEPTH / 2 + 0.001
    cuts.append(box_at(iw, ih, POCKET_DEPTH, -52.0, y_front, up_z))
    cuts.append(box_at(gw, gh, POCKET_DEPTH, 52.0, y_front, up_z))
    # Cable exits for both.
    cuts.append(box_at(8.0, 4.0, PLATE_Z * 4, -52.0, y_front + ih / 2 + 1.0, 0))
    cuts.append(box_at(8.0, 4.0, PLATE_Z * 4, 52.0, y_front + gh / 2 + 1.0, 0))

    # --- Screw slots for every board pocket -------------------------------
    # Elongated along X. Breakout vendors put mounting holes in different
    # places on nominally identical boards, and a round hole 1.5 mm off is a
    # board that will not bolt down. A 3 mm slot absorbs that.
    def board_slots(px, py, pw, ph):
        for sy in (-1, 1):
            cy = py + sy * (ph / 2 - 2.8)
            for dx in (-BOARD_SLOT_LEN / 2, BOARD_SLOT_LEN / 2):
                cuts.append(cyl_at(BOARD_SCREW_D, PLATE_Z * 4, px + dx, cy, 0))
            cuts.append(box_at(BOARD_SLOT_LEN, BOARD_SCREW_D, PLATE_Z * 4, px, cy, 0))
    board_slots(-BASELINE / 2, y_front, tw, th)      # thermal
    board_slots(BASELINE / 2, y_front, cw, ch)       # camera
    board_slots(-52.0, y_front, iw, ih)              # IMU
    board_slots(52.0, y_front, gw, gh)               # GPS module

    # --- Pi 4 on standoffs (ports and microSD stay accessible).
    for sx in (-1, 1):
        for sy in (-1, 1):
            hx = x_pi + sx * PI_HOLE_DX / 2
            hy = y_pi + sy * PI_HOLE_DY / 2
            adds.append(cyl_at(STANDOFF_D, STANDOFF_H, hx, hy, PLATE_Z / 2 + STANDOFF_H / 2))
            cuts.append(cyl_at(PI_HOLE_D, STANDOFF_H * 3, hx, hy, PLATE_Z / 2))

    # --- Power bank: walls on three sides + zip ties. Removable for charging.
    for sx in (-1, 1):
        adds.append(box_at(WALL_T, BANK_L, WALL_H,
                           x_bank + sx * (BANK_W / 2 + WALL_T / 2), y_bank, PLATE_Z / 2 + WALL_H / 2))
    adds.append(box_at(BANK_W + 2 * WALL_T, WALL_T, WALL_H,
                       x_bank, y_bank + BANK_L / 2 + WALL_T / 2, PLATE_Z / 2 + WALL_H / 2))
    for sy in (-0.28, 0.28):
        for sx in (-1, 1):
            cuts.append(box_at(ZIPTIE_W, ZIPTIE_H, PLATE_Z * 4,
                               x_bank + sx * (BANK_W / 2 - 4.0), y_bank + sy * BANK_L, 0))

    # --- Airframe interface: M3 corners + zip-tie slots on the centreline.
    for sx in (-1, 1):
        for sy in (-1, 1):
            cuts.append(cyl_at(MOUNT_HOLE_D, PLATE_Z * 4,
                               sx * (PLATE_X / 2 - 5.0), sy * (PLATE_Y / 2 - 5.0), 0))
    for sy in (-1, 1):
        cuts.append(box_at(ZIPTIE_W, ZIPTIE_H, PLATE_Z * 4, 0, sy * (PLATE_Y / 2 - 5.0), 0))

    # --- Lightening holes in the dead strip between the Pi and the bank.
    x_gap = (x_pi + PI_W / 2 + x_bank - BANK_W / 2) / 2
    for i in range(-2, 3):
        cuts.append(cyl_at(LIGHTEN_D, PLATE_Z * 4, x_gap, y_bank + i * 17.0, 0))

    # --- Ventilation ---------------------------------------------------------
    # A Pi 4 throttles at 80 C and this one runs TFLite inference, camera
    # capture and I2C polling at once, in Kolkata ambient. A solid plate 6 mm
    # under the board would trap a pocket of still hot air exactly where the
    # BCM2711 dumps its heat, so the whole Pi footprint is a grille.
    #
    # Slots run front-to-back: under a rotor the airflow is downward, and
    # front-to-back slots also match forward-flight airflow, so the tray vents
    # in both regimes. 4 mm slots on an 8.5 mm pitch keeps the ribs wide enough
    # that the plate stays stiff in bending.
    for i in range(-4, 5):
        cuts.append(box_at(VENT_W, VENT_L, PLATE_Z * 4, x_pi + i * VENT_PITCH, y_pi, 0))

    # The bank discharges at up to 3 A and warms up too; it sits on ribs rather
    # than flat on plastic so its underside can breathe.
    for i in range(-3, 4):
        cuts.append(box_at(VENT_W, BANK_L * 0.62, PLATE_Z * 4,
                           x_bank + i * (VENT_PITCH + 2.0), y_bank, 0))

    # --- microSD access ----------------------------------------------------
    # The Pi 4's card slot is on the UNDERSIDE of a short edge, and the board
    # sits only 6 mm above the plate. Without a notch you can see the card but
    # cannot get a fingernail to it -- and re-flashing means unbolting the Pi.
    # Notched at BOTH short ends so the Pi can be fitted either way round, and
    # both notches add vent area regardless.
    for sy in (-1, 1):
        cuts.append(box_at(SD_NOTCH_W, SD_NOTCH_L, PLATE_Z * 4,
                           x_pi, y_pi + sy * (PI_L / 2 - SD_NOTCH_L / 2 + 1.0), 0))

    # --- Cable tie-downs ---------------------------------------------------
    # Pairs of slots along the route from the sensor band to the Pi's GPIO
    # corner, and beside the power bank for its USB-C lead. Loose wiring on a
    # rotorcraft walks into a propeller or unseats a Dupont jumper mid-flight;
    # this is the cheapest possible fix for both.
    for ty in (y_front + 22.0, y_pi - 18.0, y_pi + 20.0):
        for sx in (-1, 1):
            cuts.append(box_at(TIE_W, TIE_H, PLATE_Z * 4, x_pi + sx * 33.0, ty, 0))
    for ty in (y_bank - 26.0, y_bank + 26.0):
        cuts.append(box_at(TIE_W, TIE_H, PLATE_Z * 4, x_bank - BANK_W / 2 - 5.5, ty, 0))

    body = trimesh.boolean.union([plate] + adds)
    body = trimesh.boolean.difference([body] + cuts)
    if isinstance(body, list):
        body = body[0]

    vol_cm3 = float(body.volume) / 1000.0
    tray_g = vol_cm3 * PETG_DENSITY * PRINT_FILL
    t_ifov = math.radians(55.0) / 32

    stats = {
        "plate_mm": [round(PLATE_X, 1), round(PLATE_Y, 1), PLATE_Z],
        "max_height_mm": round(PLATE_Z / 2 + max(STANDOFF_H, WALL_H) + BANK_H, 1),
        "watertight": bool(body.is_watertight),
        "tray_volume_cm3": round(vol_cm3, 2),
        "tray_mass_g_petg": round(tray_g, 1),
        "payload_total_g": round(tray_g + BANK_MASS_G + PI_MASS_G + IMU_MASS_G + GPS_MASS_G
                                 + GPS_ANT_MASS_G + 8 + 20, 1),
        "baseline_mm": BASELINE,
        "parallax_thermal_px_at_20m": round((BASELINE / 1000.0 / 20.0) / t_ifov, 3),
    }
    return body, stats


def build_gps_mast():
    """Raised pad for the NEO-6M ceramic patch antenna.

    A SEPARATE part, and that is the whole point. The antenna is the one
    component that must NOT sit in the payload:

    * It needs sky view. Everything else on this tray looks down.
    * A Raspberry Pi 4 is a broadband RF noise source -- the HDMI serialiser in
      particular radiates right across the L1 band at 1575 MHz. Sitting a GPS
      patch next to it costs satellites and fix quality, and the symptom is a
      slow or wandering fix rather than an obvious failure.

    Standard drone practice is a mast, so this bolts to the airframe above and
    behind the payload, not to the tray. Printed as a constant C-section
    extruded vertically, so every layer is identical and it needs no supports.
    """
    foot_l, riser_h, pad_l, t, width = 30.0, 45.0, 32.0, 3.0, 32.0
    parts = [
        box_at(foot_l, width, t, foot_l / 2, 0, t / 2),                  # foot
        box_at(t, width, riser_h, t / 2, 0, t + riser_h / 2),            # riser
        box_at(pad_l, width, t, pad_l / 2, 0, t + riser_h + t / 2),      # antenna pad
    ]
    body = trimesh.boolean.union(parts)
    cuts = []
    for sx in (0.30, 0.72):                                              # foot: M3 to airframe
        cuts.append(cyl_at(MOUNT_HOLE_D, t * 4, foot_l * sx, 0, t / 2))
    for sy in (-1, 1):                                                   # pad: zip-tie the patch
        cuts.append(box_at(ZIPTIE_W, ZIPTIE_H, t * 4,
                           pad_l * 0.62, sy * (width / 2 - 4.0), t + riser_h + t / 2))
    body = trimesh.boolean.difference([body] + cuts)
    if isinstance(body, list):
        body = body[0]
    return body, {
        "gps_mast_mm": [round(pad_l, 1), round(width, 1), round(t * 2 + riser_h, 1)],
        "gps_mast_g_petg": round(float(body.volume) / 1000.0 * PETG_DENSITY * PRINT_FILL, 1),
        "watertight": bool(body.is_watertight),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mount")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    body, stats = build()
    stl = out / "saresq_payload_tray.stl"
    body.export(stl)
    stats["stl"] = str(stl)
    mast, mstats = build_gps_mast()
    mast_stl = out / "saresq_gps_mast.stl"
    mast.export(mast_stl)
    stats.update(mstats)
    stats["gps_mast_stl"] = str(mast_stl)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if not stats["watertight"]:
        raise SystemExit("mesh is NOT watertight -- a print shop will reject it")


if __name__ == "__main__":
    main()
