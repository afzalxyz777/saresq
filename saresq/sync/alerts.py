"""Tier-1 alert packets: the smallest useful thing to put on a radio.

A SiK telemetry link shares ~16 kbps of usable goodput with MAVLink. That is
hopeless for imagery but generous for 28 bytes, so the instant a track crosses
into HIGH/MEDIUM the operator gets a map pin -- position, confidence, class --
while the crops are still queued behind it.

Position is carried as degrees x 1e7 in an int32 rather than a float32.
float32 has ~7 significant digits, and 22.5731530 needs 9; encoding Kolkata
latitudes as float32 would quantise to roughly 1-2 m, which is the same order
as the whole geolocation CEP. int32 degrees-e7 is exact to 1.1 cm.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"SQ"
VERSION = 1

# <2s B I I i i H B B B h H  -- no padding, little-endian
_FMT = "<2sBIIiiHBBBhH"
ALERT_BYTES = struct.calcsize(_FMT)  # 28

CLASS_CODES = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
DECISION_CODES = {"LOG_AND_RESUME": 0, "REOBSERVE_LOWER": 1, "CONFIRM": 2, "REJECT": 3}
_CLASS_NAMES = {v: k for k, v in CLASS_CODES.items()}
_DECISION_NAMES = {v: k for k, v in DECISION_CODES.items()}


def crc16_ccitt(data: bytes, seed: int = 0xFFFF) -> int:
    crc = seed
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass
class Alert:
    target_id: int
    t_s: int
    lat: float
    lon: float
    p_final: float
    class_: str
    decision: str
    n_passes: int
    alt_m: float


def pack_alert(a: Alert) -> bytes:
    body = struct.pack(
        _FMT[:-1],  # everything but the trailing CRC field
        MAGIC, VERSION,
        a.target_id & 0xFFFFFFFF,
        a.t_s & 0xFFFFFFFF,
        int(round(a.lat * 1e7)),
        int(round(a.lon * 1e7)),
        max(0, min(65535, int(round(a.p_final * 65535)))),
        CLASS_CODES[a.class_],
        DECISION_CODES[a.decision],
        max(0, min(255, int(a.n_passes))),
        int(round(a.alt_m * 10)),
    )
    return body + struct.pack("<H", crc16_ccitt(body))


def unpack_alert(buf: bytes) -> Alert:
    if len(buf) != ALERT_BYTES:
        raise ValueError(f"alert must be {ALERT_BYTES} bytes, got {len(buf)}")
    if crc16_ccitt(buf[:-2]) != struct.unpack("<H", buf[-2:])[0]:
        raise ValueError("alert CRC mismatch")
    magic, ver, tid, t_s, lat_e7, lon_e7, p_q, cls, dec, n, alt_dm, _ = struct.unpack(_FMT, buf)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported alert version {ver}")
    return Alert(
        target_id=tid, t_s=t_s, lat=lat_e7 / 1e7, lon=lon_e7 / 1e7,
        p_final=p_q / 65535, class_=_CLASS_NAMES[cls], decision=_DECISION_NAMES[dec],
        n_passes=n, alt_m=alt_dm / 10,
    )


def alert_from_target(row: dict, t_s: int, alt_m: float = 0.0) -> Alert:
    """Build an alert from a `targets` row."""
    return Alert(
        target_id=row["target_id"], t_s=t_s,
        lat=row["lat"] or 0.0, lon=row["lon"] or 0.0,
        p_final=row["p_final"] or 0.0,
        class_=row["class"] or "LOW",
        decision=row["decision"] or "LOG_AND_RESUME",
        n_passes=row["n_passes"] or 0, alt_m=alt_m,
    )
