"""The simulated world (Section 15.3): random survivors, distractors, ambient band."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONCEALMENT_PROBS = {"open": 0.4, "partial": 0.4, "heavy": 0.2}


@dataclass
class Person:
    x: float
    y: float
    concealment: str  # open, partial, heavy


@dataclass
class Distractor:
    x: float
    y: float


@dataclass
class World:
    width: float
    height: float
    survivors: list[Person] = field(default_factory=list)
    distractors: list[Distractor] = field(default_factory=list)
    ambient_band: str = "below30"  # or "above30"


def generate_world(
    rng: np.random.Generator,
    width: float = 200.0,
    height: float = 150.0,
    n_survivors: int = 5,
    n_distractors: int = 8,
) -> World:
    labels = list(CONCEALMENT_PROBS.keys())
    probs = list(CONCEALMENT_PROBS.values())
    survivors = [
        Person(
            x=float(rng.uniform(0, width)),
            y=float(rng.uniform(0, height)),
            concealment=str(rng.choice(labels, p=probs)),
        )
        for _ in range(n_survivors)
    ]
    distractors = [
        Distractor(x=float(rng.uniform(0, width)), y=float(rng.uniform(0, height)))
        for _ in range(n_distractors)
    ]
    ambient_band = str(rng.choice(["below30", "above30"]))
    return World(width, height, survivors, distractors, ambient_band)
