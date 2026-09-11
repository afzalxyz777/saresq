# Raspberry Pi bring-up

Payload compute for the SaResQ airframe: Pi 4, MLX90640 thermal array, Pi Camera
v2, NEO-6M GPS, MPU-6050 IMU. Headless — there is no monitor on a drone.

Written against **Raspberry Pi Imager 2.0.11.1** and **Raspberry Pi OS Bookworm
(64-bit, Lite)**.

## 1. Flash the card

Imager 2.0 replaced the old gear icon with a wizard. If you are following an
older guide, this is the step it gets wrong: **there is no ⚙️ any more**, and
the "App options" button at the bottom left is Imager's own preferences, not
this. The sidebar is:

```
Device → OS → Storage → Customisation → Writing → Done
```

* **Device** — Raspberry Pi 4
* **OS** — Raspberry Pi OS (other) → **Raspberry Pi OS Lite (64-bit)**.
  Lite because a desktop on an airframe wastes RAM and SD write cycles.
  64-bit because the TFLite and OpenCV wheels are properly supported on
  `aarch64`, which matters the moment the detector comes back from training.
* **Storage** — the microSD. Check the size matches the card; Imager erases
  whatever you point it at.
* **Customisation** — the step that makes the Pi reachable:
  * hostname `saresq`
  * enable SSH (password auth is fine on a lab network; a key is better)
  * username + password — write them down, there is no recovery
  * Wi-Fi SSID + password, **wireless LAN country `IN`** (the radio stays off
    without a country set)
  * locale / timezone

Then Write. It verifies afterwards — let it finish.

## 2. Boot configuration

Imager cannot set these, so apply them while the card is still in the Mac. It
mounts as `bootfs`.

```bash
python tools/pi_bootfs_config.py --check   # see what it would do
python tools/pi_bootfs_config.py           # apply
```

Do not hand-edit `config.txt` in TextEdit: it will offer to save as RTF or as
`config.txt.txt`, and the Pi then silently ignores the file and boots with no
I²C. That failure looks exactly like broken wiring and costs an afternoon.

What it sets, and why:

| Line | Why |
|---|---|
| `dtparam=i2c_arm=on` | I²C on GPIO 2/3 — MLX90640 and MPU-6050 |
| `dtparam=i2c_arm_baudrate=400000` | the MLX90640 moves 1544 bytes per frame; 100 kHz cannot keep up |
| `enable_uart=1` | hardware serial on GPIO 14/15 for the NEO-6M |
| `dtoverlay=disable-bt` | gives the GPS the stable PL011 UART. The mini-UART's baud rate follows the CPU clock and drifts under load |
| `camera_auto_detect=1` | CSI camera (already Bookworm's default; pinned so an edit cannot lose it) |

And one thing the obvious instructions all miss:

**It removes `console=serial0,115200` from `cmdline.txt`.** Raspberry Pi OS puts
a serial login console on the same UART the GPS needs. Leave it and `getty` and
the NEO-6M both hold `/dev/ttyAMA0`: you get garbage NMEA or silence, which is
indistinguishable from a dead module or swapped TX/RX. `console=tty1` is kept —
that is the HDMI console and is harmless.

Then eject properly. FAT32 writes sit in the page cache until unmount, so
pulling the card here is a real way to produce a card that fails on first boot:

```bash
diskutil eject /Volumes/bootfs
```

## 3. First boot

Card into the slot on the underside, power on. Red LED solid immediately (power
good). Green flickers 30–90 s while it boots and resizes the root filesystem,
then settles.

Power matters: the Pi 4 wants **5 V at 3 A**. A laptop port or a bus-powered hub
undervolts it, and the symptom is SD corruption that looks like a bad flash.

```bash
ssh <your-username>@saresq.local
```

If `saresq.local` does not resolve, the router is blocking mDNS — find the IP on
the router's client list (hostname `saresq`) and use that.

If SSH times out entirely, the Pi never joined Wi-Fi: the SSID or password in
step 1 was wrong (both case-sensitive), or the country was left unset.

## 4. Verify the boot took

```bash
vcgencmd get_throttled          # want throttled=0x0
grep -E "i2c|uart|camera" /boot/firmware/config.txt
cat /boot/firmware/cmdline.txt  # console=serial0 must be GONE
ls -l /dev/ttyAMA0 /dev/i2c-1
```

`throttled=0x0` means the supply is holding up. Anything else is the power
supply, not the software — fix it before going further, because undervoltage
corrupts cards.

## 5. Finish the serial handover and install tools

`dtoverlay=disable-bt` frees the UART, but the Bluetooth-to-UART service is
still enabled and will hold the port:

```bash
sudo systemctl disable --now hciuart
sudo apt update
sudo apt install -y i2c-tools python3-venv python3-pip libatlas-base-dev
```

`i2c-tools` is what provides `i2cdetect` — without it the command simply does
not exist, which reads as a broken I²C bus.

## 6. Sensor checks, as each one is wired

Power the Pi **off** before touching the GPIO header.

```bash
sudo i2cdetect -y 1        # empty until something is wired
```

Expected addresses once connected:

| Device | Address |
|---|---|
| MLX90640 thermal array | `0x33` |
| MPU-6050 IMU | `0x68` (`0x69` if AD0 is pulled high) |

Camera:

```bash
rpicam-hello --list-cameras     # Bookworm; it was libcamera-hello on Bullseye
```

GPS — the module needs a clear view of sky and can take several minutes for a
first fix from cold:

```bash
sudo cat /dev/ttyAMA0           # expect NMEA sentences, $GPGGA / $GPRMC
```

Garbage characters here usually mean a baud mismatch (NEO-6M defaults to 9600)
rather than a wiring fault. Nothing at all means TX/RX are swapped, or the
serial console was not removed in step 2.

## 7. Python environment

```bash
python3 -m venv ~/saresq-venv
source ~/saresq-venv/bin/activate
pip install --upgrade pip
```

Install the payload dependencies from the repo once it is cloned onto the Pi.
Prefer piwheels-backed wheels; building NumPy or OpenCV from source on a Pi 4
takes hours.

## Files

| Path | What |
|---|---|
| `tools/pi_bootfs_config.py` | applies section 2 to a freshly-flashed card, idempotently |
| `docs/colab_training.md` | the training runbook this hardware is waiting on |
