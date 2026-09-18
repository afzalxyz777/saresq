"""Thermal-frame processing that goes beyond a single frame.

`drizzle` implements multi-frame super-resolution for the MLX90640's 32x24
array, after Fruchter & Hook's variable-pixel linear reconstruction (PASP 114,
144, 2002) as used on the Hubble Space Telescope.
"""
from saresq.thermal.drizzle import Drizzle, DrizzleResult, estimate_shift

__all__ = ["Drizzle", "DrizzleResult", "estimate_shift"]
