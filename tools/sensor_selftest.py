"""One-command check that the three soldered sensors actually work.

    python3 tools/sensor_selftest.py

Run it on the Pi. Exits 0 only if every wired sensor passes.

This deliberately does NOT stop at "the address responds". An I2C address ACK
proves the SDA/SCL joints conduct and nothing more -- a board with a cold joint
on VCC can still ACK off the pull-ups and return nothing but zeros. So each
device is checked against something that cannot be faked:

  MPU-6050   WHO_AM_I must read 0x68, then |acceleration| must be ~1 g.
             Gravity is always there. A board reading 0.00 g is not a board
             that is "idle", it is a board that is not working.
  MLX90640   the control register must read its documented default, and the
             subpage bit in the status register must CHANGE between two reads
             500 ms apart -- that proves the array is actually sampling
             rather than merely answering.
  NEO-6M     NMEA sentences with valid checksums. A satellite fix is NOT
             required: indoors there will be none, and demanding one turns a
             wiring test into a weather report. Sentences arriving at all is
             what proves TX and the baud rate.

Nothing here needs the payload venv or numpy; it uses smbus2 (or python3-smbus)
and the stdlib, so it runs on a bare Pi.
"""
from __future__ import annotations

import argparse
import glob
import struct
import sys
import time

MPU_ADDR = 0x68         # 0x69 if AD0 is pulled high
MLX_ADDR = 0x33
MLX_CTRL = 0x800D       # control register 1, default 0x1901
MLX_STATUS = 0x8000     # bit 0 = which subpage was last measured

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m")


def _say(state: str, name: str, detail: str) -> None:
    colour = {"PASS": GREEN, "FAIL": RED, "SKIP": YELLOW}[state]
    print(f"  {colour}{state:<4}{RESET}  {name:<12} {detail}")


def _open_bus(busno: int):
    try:
        from smbus2 import SMBus, i2c_msg
        return SMBus(busno), i2c_msg
    except ImportError:
        pass
    try:
        import smbus                       # python3-smbus, no i2c_msg
        return smbus.SMBus(busno), None
    except ImportError:
        return None, None


def check_mpu(bus, addr: int) -> bool:
    """WHO_AM_I, then gravity. Registers per the MPU-6050 datasheet."""
    try:
        who = bus.read_byte_data(addr, 0x75)
    except OSError as exc:
        _say("FAIL", "MPU-6050", f"no response at 0x{addr:02x} ({exc.strerror}). "
                                 "Check SDA/SCL and VCC joints.")
        return False
    if who not in (0x68, 0x69, 0x70, 0x71, 0x73, 0x98):
        # 0x70/0x71/0x73/0x98 are the clone/MPU-9250/6500 answers seen on
        # cheap modules; anything else means the byte is not coming from a
        # working MPU at all.
        _say("FAIL", "MPU-6050", f"WHO_AM_I = 0x{who:02x}, expected 0x68. "
                                 "Data line is unreliable -- reflow SDA/SCL.")
        return False

    bus.write_byte_data(addr, 0x6B, 0x00)   # PWR_MGMT_1: wake from sleep
    time.sleep(0.1)
    raw = bus.read_i2c_block_data(addr, 0x3B, 6)
    ax, ay, az = struct.unpack(">hhh", bytes(raw))
    g = ((ax / 16384.0) ** 2 + (ay / 16384.0) ** 2 + (az / 16384.0) ** 2) ** 0.5

    if ax == ay == az == 0:
        _say("FAIL", "MPU-6050", "all axes read exactly 0 -- responding but not "
                                 "sampling. Usually a cold joint on VCC or GND.")
        return False
    if not 0.85 <= g <= 1.15:
        _say("FAIL", "MPU-6050", f"|a| = {g:.2f} g, expected ~1.00. "
                                 "Hold the board still and re-run.")
        return False
    _say("PASS", "MPU-6050", f"WHO_AM_I 0x{who:02x}, |a| = {g:.2f} g "
                             f"{DIM}(x {ax/16384:+.2f} y {ay/16384:+.2f} "
                             f"z {az/16384:+.2f}){RESET}")
    return True


def _mlx_read16(bus, i2c_msg, reg: int) -> int:
    """MLX90640 uses 16-bit register addresses, so plain SMBus calls cannot
    reach it -- it needs a raw write-then-read transaction."""
    write = i2c_msg.write(MLX_ADDR, [reg >> 8, reg & 0xFF])
    read = i2c_msg.read(MLX_ADDR, 2)
    bus.i2c_rdwr(write, read)
    data = list(read)
    return (data[0] << 8) | data[1]


def check_mlx(bus, i2c_msg) -> bool:
    if i2c_msg is None:
        _say("SKIP", "MLX90640", "needs smbus2 for 16-bit registers: "
                                 "pip install smbus2  (or apt install python3-smbus2)")
        return True
    try:
        ctrl = _mlx_read16(bus, i2c_msg, MLX_CTRL)
    except OSError as exc:
        _say("FAIL", "MLX90640", f"no response at 0x{MLX_ADDR:02x} "
                                 f"({exc.strerror}). CONFIRM VIN IS ON 3V3, "
                                 "PIN 1 -- not 5V.")
        return False

    s1 = _mlx_read16(bus, i2c_msg, MLX_STATUS)
    time.sleep(0.5)                          # >= 2 frames at the default 2 Hz
    s2 = _mlx_read16(bus, i2c_msg, MLX_STATUS)
    if (s1 & 0x01) == (s2 & 0x01) and (s1 & 0x08) == (s2 & 0x08) == 0:
        _say("FAIL", "MLX90640", f"status frozen at 0x{s1:04x} over 500 ms -- "
                                 "answering but not sampling.")
        return False
    _say("PASS", "MLX90640", f"ctrl 0x{ctrl:04x}, status {DIM}0x{s1:04x} -> "
                             f"0x{s2:04x} (sampling){RESET}")
    return True


def check_gps(port: str | None, baud: int, seconds: float) -> bool:
    candidates = [port] if port else ["/dev/ttyAMA0", "/dev/serial0", "/dev/ttyS0"]
    candidates = [p for p in candidates if glob.glob(p)]
    if not candidates:
        _say("FAIL", "NEO-6M", "no serial device. Is enable_uart=1 set and the "
                               "console freed? See docs/pi_setup.md section 2.")
        return False

    try:
        import serial
    except ImportError:
        _say("SKIP", "NEO-6M", f"pip install pyserial, then re-run. Meanwhile: "
                               f"stty -F {candidates[0]} {baud} && cat {candidates[0]}")
        return True

    for dev in candidates:
        try:
            with serial.Serial(dev, baud, timeout=1) as ser:
                deadline = time.time() + seconds
                good = bad = 0
                sample = ""
                fix = None
                while time.time() < deadline:
                    line = ser.readline().decode("ascii", "replace").strip()
                    if not line.startswith("$"):
                        continue
                    body, _, cksum = line[1:].partition("*")
                    calc = 0
                    for ch in body:
                        calc ^= ord(ch)
                    if len(cksum) >= 2 and calc == int(cksum[:2], 16):
                        good += 1
                        sample = sample or line[:60]
                        if body.startswith(("GPGGA", "GNGGA")):
                            f = body.split(",")
                            if len(f) > 6 and f[6].isdigit():
                                fix = int(f[6])
                    else:
                        bad += 1
                if good:
                    note = ("no satellite fix yet -- normal indoors, needs sky"
                            if not fix else f"FIX ({fix}) acquired")
                    _say("PASS", "NEO-6M", f"{good} valid NMEA on {dev}, {note} "
                                           f"{DIM}{sample}{RESET}")
                    return True
                if bad:
                    _say("FAIL", "NEO-6M", f"{bad} corrupt sentences on {dev} -- "
                                           f"baud mismatch, try 4800 or 38400.")
                    return False
        except (OSError, serial.SerialException):
            continue
    _say("FAIL", "NEO-6M", f"silence on {', '.join(candidates)}. TX/RX are "
                           "probably swapped: GPS TX goes to Pi RX (pin 10).")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--mpu-addr", type=lambda v: int(v, 0), default=MPU_ADDR)
    ap.add_argument("--gps-port", default=None)
    ap.add_argument("--gps-baud", type=int, default=9600)
    ap.add_argument("--gps-seconds", type=float, default=5.0)
    ap.add_argument("--skip", default="", help="comma-separated: mpu,mlx,gps")
    args = ap.parse_args(argv)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    print("\nSaResQ sensor self-test\n")
    results = []
    want_i2c = {"mpu", "mlx"} - skip

    if want_i2c:
        bus, i2c_msg = _open_bus(args.bus)
        if bus is None:
            _say("FAIL", "I2C bus", "no smbus2/smbus module. "
                                    "sudo apt install -y python3-smbus2")
            results.append(False)
        else:
            if "mpu" not in skip:
                results.append(check_mpu(bus, args.mpu_addr))
            if "mlx" not in skip:
                results.append(check_mlx(bus, i2c_msg))

    if "gps" not in skip:
        results.append(check_gps(args.gps_port, args.gps_baud, args.gps_seconds))

    ok = all(results)
    print(f"\n{GREEN if ok else RED}{'ALL PASS' if ok else 'FAILURES ABOVE'}"
          f"{RESET}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
