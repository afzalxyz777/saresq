#!/usr/bin/env bash
# Rebuild a freshly-flashed Pi into the SaResQ payload, in one command.
#
#     bash ~/pi_bootstrap.sh
#
# Everything here was learned the hard way on 2026-09-11/12; the comments say
# why, so a future reflash does not rediscover it.
set -euo pipefail

VENV="$HOME/saresq-venv"
LOG="$HOME/bootstrap.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== SaResQ bootstrap $(date) ==="

echo
echo "--- 1/5 apt packages ---"
sudo apt-get update -q
# i2c-tools: without it `i2cdetect` simply does not exist, which reads as a
#   dead bus rather than a missing package.
# swig + python3-dev: lgpio (pulled in by Blinka) builds from source and fails
#   with "command 'swig' failed" without them. liblgpio-dev alone is NOT enough.
# libatlas-base-dev is deliberately absent: it does not exist in Debian 13
# (trixie) and is only needed when numpy is built from source. The wheels used
# below are prebuilt, so nothing wants it.
sudo apt-get install -y -q \
    i2c-tools python3-venv python3-dev python3-smbus2 \
    swig liblgpio-dev python3-lgpio

echo
echo "--- 2/5 enable I2C and the UART ---"
CFG=/boot/firmware/config.txt
sudo raspi-config nonint do_i2c 0            # 0 = enable
sudo raspi-config nonint do_serial_hw 0      # UART on, for the GPS
sudo raspi-config nonint do_serial_cons 1    # serial LOGIN console OFF
# 400 kHz: at the 100 kHz default a full MLX90640 frame cannot be clocked out
# fast enough and frames tear across the subpage boundary.
grep -q '^dtparam=i2c_arm_baudrate' "$CFG" \
  || echo 'dtparam=i2c_arm_baudrate=400000' | sudo tee -a "$CFG" >/dev/null
# The Bluetooth-to-UART service holds the port the GPS needs even after
# disable-bt, so it is stopped explicitly.
sudo systemctl disable --now hciuart 2>/dev/null || true

echo
echo "--- 3/5 python environment ---"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel
# piwheels serves prebuilt ARM wheels; without it numpy and OpenCV compile from
# source and take hours on a Pi 4.
"$VENV/bin/pip" install -q \
    numpy opencv-python-headless ai-edge-litert pyserial smbus2 pyyaml \
    adafruit-blinka adafruit-circuitpython-mlx90640

echo
echo "--- 4/5 verify the imports that matter ---"
"$VENV/bin/python3" - <<'PY'
import importlib, sys
need = ["numpy", "cv2", "yaml", "serial", "smbus2",
        "board", "busio", "adafruit_mlx90640", "ai_edge_litert"]
bad = []
for m in need:
    try:
        importlib.import_module(m)
        print(f"  ok    {m}")
    except Exception as exc:
        bad.append(m)
        print(f"  FAIL  {m}: {exc}")
sys.exit(1 if bad else 0)
PY

echo
echo "--- 5/5 done ---"
echo "I2C bus:"
sudo i2cdetect -y 1 || true
echo
echo "A REBOOT is required for the I2C/UART changes to take effect."
echo "Then:  ~/saresq-venv/bin/python3 ~/saresq/tools/sensor_selftest.py"
echo "Log:   $LOG"
