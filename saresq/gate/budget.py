"""Ambient-adaptive region budget and RGB fallback trigger (Section 5.4)."""
from __future__ import annotations


def trust(z_max: float, z_lo: float = 1.5, z_hi: float = 4.0) -> float:
    """c in [0, 1]: how much the gate is currently trusted."""
    if z_hi <= z_lo:
        raise ValueError("z_hi must exceed z_lo")
    c = (z_max - z_lo) / (z_hi - z_lo)
    return min(max(c, 0.0), 1.0)


def crop_budget(z_max: float, z_lo: float = 1.5, z_hi: float = 4.0, n_min: int = 2, n_max: int = 6) -> int:
    """Number of crops the detector is allowed this frame.

    Strong contrast (c near 1) -> trust the gate, crop few (n_min).
    Weak contrast (c near 0) -> widen the net, crop many (n_max).
    """
    c = trust(z_max, z_lo, z_hi)
    return round(n_min + (n_max - n_min) * (1 - c))


class FallbackTrigger:
    """Tracks consecutive zero-contrast frames to arm the RGB full-frame fallback."""

    def __init__(self, enable_after_frames: int = 3, z_lo: float = 1.5, z_hi: float = 4.0):
        self.enable_after_frames = enable_after_frames
        self.z_lo = z_lo
        self.z_hi = z_hi
        self._streak = 0

    def update(self, z_max: float) -> bool:
        """Feed one frame's z_max; returns True once the fallback should engage."""
        c = trust(z_max, self.z_lo, self.z_hi)
        self._streak = self._streak + 1 if c == 0.0 else 0
        return self._streak >= self.enable_after_frames

    def reset(self) -> None:
        self._streak = 0
