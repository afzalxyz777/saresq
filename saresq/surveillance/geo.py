"""Local tangent plane. Filtering happens in metres, never in degrees.

Over a survey area a kilometre across, a flat east-north-up plane pinned at the
launch point is accurate to well under the GPS noise, and it keeps the tracker
free of any latitude-dependent scaling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# WGS-84 meridional and transverse radii evaluated once at the AOI latitude.
# Recomputing per call would be more correct and less honest: the error over
# 1 km is far below a millimetre and this is called thousands of times a scan.
_A = 6378137.0
_E2 = 6.69437999014e-3


@dataclass(frozen=True)
class LocalFrame:
    """East-north plane with its origin at (lat0, lon0)."""

    lat0: float
    lon0: float

    @property
    def _scales(self) -> tuple[float, float]:
        s = math.sin(math.radians(self.lat0))
        w = math.sqrt(1.0 - _E2 * s * s)
        m_per_deg_lat = math.pi * _A * (1.0 - _E2) / (180.0 * w ** 3)
        m_per_deg_lon = math.pi * _A * math.cos(math.radians(self.lat0)) / (180.0 * w)
        return m_per_deg_lat, m_per_deg_lon

    def to_en(self, lat: float, lon: float) -> tuple[float, float]:
        mlat, mlon = self._scales
        return (lon - self.lon0) * mlon, (lat - self.lat0) * mlat

    def to_ll(self, e: float, n: float) -> tuple[float, float]:
        mlat, mlon = self._scales
        return self.lat0 + n / mlat, self.lon0 + e / mlon


#: IUGG mean earth radius. Haversine is a spherical formula, so it takes the
#: mean radius, not the equatorial one -- using _A here reads about 0.3% long
#: at this latitude, which is small but is a second, disagreeing earth model
#: sitting next to LocalFrame's ellipsoidal one.
_R_MEAN = 6371008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on a sphere.

    Used for link range and leg lengths, where a tenth of a percent is far
    below anything that matters. Anything being *filtered* goes through
    LocalFrame instead, which is ellipsoidal.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R_MEAN * math.asin(min(1.0, math.sqrt(a)))
