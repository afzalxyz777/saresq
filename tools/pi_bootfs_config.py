"""Apply SaResQ's boot configuration to a freshly-flashed Raspberry Pi card.

    python tools/pi_bootfs_config.py          # apply
    python tools/pi_bootfs_config.py --check  # report only, change nothing

Run this on the Mac after Raspberry Pi Imager finishes, while the card is still
inserted. It edits the FAT32 boot partition that Imager leaves mounted as
`bootfs`.

Why a script rather than "open config.txt and add five lines":

  * TextEdit on macOS will happily save config.txt as RTF, or as config.txt.txt,
    and the Pi then ignores it entirely and boots with no I2C. That failure is
    silent and looks like broken wiring.
  * It is idempotent, so re-running after a re-flash is safe and re-running on
    an already-configured card does nothing.
  * It handles cmdline.txt, which the hand-written instructions miss -- see
    SERIAL_CONSOLE below. That one is the difference between a GPS that works
    and a GPS that returns garbage.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys

# --- config.txt --------------------------------------------------------------
# Each entry is (regex that recognises the setting, exact line to write, why).
SETTINGS: list[tuple[str, str, str]] = [
    (r"^\s*dtparam=i2c_arm=", "dtparam=i2c_arm=on",
     "I2C on GPIO 2/3 for the MLX90640 thermal array and the MPU-6050 IMU"),
    (r"^\s*dtparam=i2c_arm_baudrate=", "dtparam=i2c_arm_baudrate=400000",
     "400 kHz fast mode; the MLX90640 moves 1544 bytes per frame and stalls at 100 kHz"),
    (r"^\s*enable_uart=", "enable_uart=1",
     "hardware serial on GPIO 14/15 for the NEO-6M GPS"),
    (r"^\s*dtoverlay=disable-bt", "dtoverlay=disable-bt",
     "hands the stable PL011 UART to the GPS instead of Bluetooth; the mini-UART's "
     "baud rate follows the CPU clock and drifts under load"),
    (r"^\s*camera_auto_detect=", "camera_auto_detect=1",
     "Pi Camera v2 over CSI (already default on Bookworm; pinned so it survives edits)"),
]

HEADER = "# --- SaResQ payload (added by tools/pi_bootfs_config.py) ---"

# --- cmdline.txt -------------------------------------------------------------
# Raspberry Pi OS puts a serial login console on the same UART the GPS needs.
# getty and the NEO-6M then both hold the port: the GPS reads as garbage NMEA,
# or as nothing at all, which is indistinguishable from a dead module or a
# miswired TX/RX. Removing this is not optional if the GPS is going on GPIO.
SERIAL_CONSOLE = re.compile(r"\s*console=(serial0|ttyAMA0|ttyS0),\d+")


def find_bootfs() -> pathlib.Path | None:
    """The Imager-written boot partition, by content rather than by name.

    Matching on the volume label alone is not enough: a card can be labelled
    `bootfs` and be anything. Requiring config.txt and cmdline.txt means we
    never write to the wrong volume.
    """
    for vol in sorted(pathlib.Path("/Volumes").glob("*")):
        try:
            if (vol / "config.txt").is_file() and (vol / "cmdline.txt").is_file():
                return vol
        except OSError:
            continue
    return None


def patch_config(path: pathlib.Path, apply: bool) -> list[str]:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    todo, present = [], []
    for pattern, line, why in SETTINGS:
        rx = re.compile(pattern, re.M)
        hit = next((l for l in lines if rx.match(l)), None)
        if hit is None:
            todo.append((line, why))
        elif hit.strip() != line:
            todo.append((line, why + f"  [replacing: {hit.strip()}]"))
            lines = [l for l in lines if not rx.match(l)]
        else:
            present.append(line)

    for line in present:
        print(f"    already set   {line}")
    for line, why in todo:
        print(f"    {'ADD ' if apply else 'WOULD ADD'}          {line}\n"
              f"                  ↳ {why}")
    if apply and todo:
        shutil.copy2(path, path.with_suffix(".txt.saresq-backup"))
        block = [""] + [HEADER] + [l for l, _ in todo] + [""]
        path.write_text("\n".join(lines + block))
    return [l for l, _ in todo]


def patch_cmdline(path: pathlib.Path, apply: bool) -> bool:
    text = path.read_text(errors="replace").strip()
    if not SERIAL_CONSOLE.search(text):
        print("    already clear  no serial login console on the GPS UART")
        return False
    new = SERIAL_CONSOLE.sub("", text)
    new = re.sub(r"\s{2,}", " ", new).strip()
    print(f"    {'REMOVE' if apply else 'WOULD REMOVE'}       "
          f"{SERIAL_CONSOLE.search(text).group().strip()}\n"
          "                  ↳ getty would otherwise fight the NEO-6M for /dev/ttyAMA0")
    if apply:
        shutil.copy2(path, path.with_suffix(".txt.saresq-backup"))
        # cmdline.txt must stay a single line -- the bootloader reads line 1 only.
        path.write_text(new + "\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--volume", help="path to the boot partition, if autodetect fails")
    args = ap.parse_args()
    apply = not args.check

    boot = pathlib.Path(args.volume) if args.volume else find_bootfs()
    if boot is None:
        print("No Raspberry Pi boot partition found under /Volumes.\n"
              "Flash the card with Raspberry Pi Imager first; it mounts as `bootfs`.\n"
              "If the card is in but not mounted, re-insert it.")
        return 1

    print(f"boot partition: {boot}")
    print("\n  config.txt")
    added = patch_config(boot / "config.txt", apply)
    print("\n  cmdline.txt")
    changed = patch_cmdline(boot / "cmdline.txt", apply)

    if args.check:
        print("\n--check: nothing was written.")
    elif added or changed:
        print("\nWritten. Originals saved alongside as *.saresq-backup.")
        print(f"Now eject properly:  diskutil eject '{boot}'")
        print("Do NOT pull the card -- FAT32 writes sit in the page cache until eject.")
    else:
        print("\nNothing to do; this card is already configured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
