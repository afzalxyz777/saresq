"""Pass G: the multi-pass confidence ledger (Section 10.7, 10.9).

Log-odds accumulation across mission passes with a correlation discount for
repeated observations from the same altitude band. Pure Python/NumPy, no
learned component here — the fusion head (Section 10.4) supplies each p_k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class PassRecord:
    pass_id: int
    alt_band: int
    p_k: float
    summary: dict = field(default_factory=dict)


class Target:
    """One physical location being evaluated across mission passes."""

    def __init__(
        self,
        lat: float,
        lon: float,
        pi_0: float = 0.20,
        w_repeat: float = 0.7,
        confirm_p: float = 0.90,
        reject_p: float = 0.10,
        max_passes: int = 3,
    ):
        self.lat, self.lon = lat, lon
        self.pi_0 = pi_0
        self.w_repeat = w_repeat
        self.confirm_p = confirm_p
        self.reject_p = reject_p
        self.max_passes = max_passes
        self.log_odds = logit(pi_0)
        self.bands_seen: set[int] = set()
        self.passes: list[PassRecord] = []

    def add_pass(self, alt_band: int, p_k: float, summary: dict | None = None) -> None:
        p_k = min(max(p_k, 1e-3), 1 - 1e-3)  # avoid infinite log-odds
        w = 1.0 if alt_band not in self.bands_seen else self.w_repeat
        self.log_odds += w * (logit(p_k) - logit(self.pi_0))
        self.bands_seen.add(alt_band)
        self.passes.append(PassRecord(len(self.passes) + 1, alt_band, p_k, summary or {}))

    @property
    def p_final(self) -> float:
        return sigmoid(self.log_odds)

    def decision(self) -> str:
        if self.p_final >= self.confirm_p:
            return "CONFIRM"
        if self.p_final <= self.reject_p:
            return "REJECT"
        if len(self.passes) >= self.max_passes:
            return "LOG_AND_RESUME"
        return "REOBSERVE_LOWER"
