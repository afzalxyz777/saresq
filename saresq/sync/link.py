"""Link abstraction plus a simulated one, so the whole sync path is testable
on a laptop with no radio, no drone and no Wi-Fi.

Frames are self-describing (kind, media_id, offset) so a receiver can
reassemble out of a lossy, resumed, interleaved stream. That framing is what
makes the transfer genuinely resumable rather than resumable-in-principle.
"""
from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field

FRAME_ALERT = 1
FRAME_MEDIA = 2

_MEDIA_HDR = "<BIQ"  # kind, media_id, offset
MEDIA_HDR_BYTES = struct.calcsize(_MEDIA_HDR)  # 13


def frame_alert(payload: bytes) -> bytes:
    return struct.pack("<B", FRAME_ALERT) + payload


def frame_media(media_id: int, offset: int, chunk: bytes) -> bytes:
    return struct.pack(_MEDIA_HDR, FRAME_MEDIA, media_id, offset) + chunk


@dataclass
class LinkStats:
    sent_bytes: int = 0
    delivered: int = 0
    dropped: int = 0
    seconds: float = 0.0


class Receiver:
    """Ground-station side: reassembles frames into whole artefacts."""

    def __init__(self) -> None:
        self.alerts: list[bytes] = []
        self.buffers: dict[int, bytearray] = {}

    def feed(self, frame: bytes) -> None:
        kind = frame[0]
        if kind == FRAME_ALERT:
            self.alerts.append(frame[1:])
            return
        _, media_id, offset = struct.unpack(_MEDIA_HDR, frame[:MEDIA_HDR_BYTES])
        body = frame[MEDIA_HDR_BYTES:]
        buf = self.buffers.setdefault(media_id, bytearray())
        if offset > len(buf):
            # A gap means a frame was lost; the agent will resend from its own
            # persisted offset, so refuse to write past the end rather than
            # silently zero-filling a hole into the middle of an image.
            return
        buf[offset:offset + len(body)] = body

    def assembled(self, media_id: int) -> bytes:
        return bytes(self.buffers.get(media_id, b""))


class Link:
    """Base class. Subclasses model a real radio or Wi-Fi association."""

    name = "link"
    mtu = 1024
    goodput_bps = 1_000_000

    def __init__(self) -> None:
        self.stats = LinkStats()

    def is_up(self) -> bool:
        return True

    def send(self, frame: bytes) -> bool:
        raise NotImplementedError

    def seconds_for(self, n_bytes: int) -> float:
        return n_bytes * 8 / self.goodput_bps if self.goodput_bps > 0 else 0.0


class SimulatedLink(Link):
    """A deterministic, deliberately unreliable link.

    up_after_bytes / down_after_bytes let a test force a drop partway through a
    large artefact and then verify the transfer resumes at exactly the right
    offset instead of restarting.
    """

    def __init__(
        self,
        goodput_bps: float = 1_000_000,
        mtu: int = 1024,
        loss: float = 0.0,
        seed: int = 0,
        down_after_bytes: int | None = None,
        name: str = "sim",
        receiver: Receiver | None = None,
    ):
        super().__init__()
        self.goodput_bps = goodput_bps
        self.mtu = mtu
        self.loss = loss
        self.rng = random.Random(seed)
        self.down_after_bytes = down_after_bytes
        self.name = name
        self.receiver = receiver if receiver is not None else Receiver()
        self._forced_down = False

    def is_up(self) -> bool:
        if self._forced_down:
            return False
        if self.down_after_bytes is not None and self.stats.sent_bytes >= self.down_after_bytes:
            return False
        return True

    def bring_up(self) -> None:
        """Operator walked closer / drone climbed: the link is back."""
        self._forced_down = False
        self.down_after_bytes = None

    def bring_down(self) -> None:
        self._forced_down = True

    def send(self, frame: bytes) -> bool:
        if not self.is_up():
            return False
        self.stats.seconds += self.seconds_for(len(frame))
        if self.rng.random() < self.loss:
            self.stats.dropped += 1
            return False
        self.stats.sent_bytes += len(frame)
        self.stats.delivered += 1
        self.receiver.feed(frame)
        return True


class TelemetryLink(SimulatedLink):
    """SiK-class radio: ~16 kbps usable after MAVLink overhead, small MTU.

    Fine for alerts, hopeless for imagery -- which is exactly why the agent
    ranks alerts ahead of everything else.
    """

    def __init__(self, **kw):
        kw.setdefault("goodput_bps", 16_000)
        kw.setdefault("mtu", 200)
        kw.setdefault("name", "sik")
        super().__init__(**kw)


class WifiLink(SimulatedLink):
    """Onboard Pi Wi-Fi to a ground station inside the survey segment.

    Max slant range across a 220x140 m segment from launch is ~260 m, at 20 m
    AGL with clear line of sight. Conservative goodput, generous MTU.
    """

    def __init__(self, **kw):
        kw.setdefault("goodput_bps", 4_000_000)
        kw.setdefault("mtu", 1400)
        kw.setdefault("name", "wifi")
        super().__init__(**kw)
