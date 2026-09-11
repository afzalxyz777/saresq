"""Deploy side of the fusion head (Section 10.4): load the JSON, dot, sigmoid.

The training half lives in ``training/train_fusion.py``. Nothing in this module
imports scikit-learn, and that is the point -- the payload computer carries a
JSON file and eighteen multiply-adds, not a model runtime.

The head is linear on purpose. A small MLP scores marginally better offline,
but this output is consumed by ``saresq.fuse.ledger``, which accumulates
``logit(p_k)`` across mission passes. A linear model in log-odds space means
the ledger's arithmetic and the head's arithmetic are the same arithmetic, so
a per-feature contribution can be read straight off the weight vector and
shown to an operator as a reason. ``explain()`` exists for exactly that.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES

FORMAT = "saresq.fusion.logreg/1"


@dataclass(frozen=True)
class FusionHead:
    weights: np.ndarray
    bias: float
    pi_0: float
    zeroed: tuple[str, ...] = ()
    metrics: dict | None = None

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "FusionHead":
        blob = json.loads(pathlib.Path(path).read_text())
        if blob.get("format") != FORMAT:
            raise ValueError(f"expected format {FORMAT!r}, got {blob.get('format')!r}")
        # Order is the contract between training and deployment. Checking it
        # costs nothing and catches the one bug that would otherwise be silent:
        # a reordered FEATURE_NAMES scrambles which weight lands on which
        # feature, producing plausible probabilities that are entirely wrong.
        if blob.get("feature_names") != FEATURE_NAMES:
            raise ValueError(
                "fusion head feature order does not match saresq.fuse.features.FEATURE_NAMES; "
                "the head must be re-exported by training/train_fusion.py"
            )
        weights = np.asarray(blob["weights"], dtype=float)
        if weights.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"expected {len(FEATURE_NAMES)} weights, got {weights.shape}")
        return cls(
            weights=weights,
            bias=float(blob["bias"]),
            pi_0=float(blob["pi_0"]),
            zeroed=tuple(blob.get("zeroed_features", ())),
            metrics=blob.get("metrics"),
        )

    def logit(self, features: np.ndarray) -> float:
        return float(np.dot(self.weights, features) + self.bias)

    def predict(self, features: np.ndarray) -> float:
        """Calibrated P(survivor | evidence) for one 18-feature vector."""
        if features.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {features.shape}")
        z = self.logit(features)
        # Branch on the sign so neither exp() overflows: exp(+800) is inf,
        # which turns a confident detection into a nan and drops it silently.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def explain(self, features: np.ndarray, top: int = 5) -> list[tuple[str, float]]:
        """The ``top`` features moving this decision, by log-odds contribution.

        Signed: positive pushed towards survivor, negative away. This is what
        the operator review queue shows next to a candidate, so the reason a
        score is high is auditable rather than asserted.
        """
        contributions = self.weights * features
        order = np.argsort(-np.abs(contributions))[:top]
        return [(FEATURE_NAMES[i], float(contributions[i])) for i in order]
