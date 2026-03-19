from dataclasses import dataclass
from typing import Optional

from senfuslib import NamedArray, AtIndex, MetaData


# -------------------------
# Tracking measurements
# -------------------------

@dataclass
class UsblMeasurement(NamedArray):
    """Azimuth, Elevation (radians)"""
    azimuth: AtIndex[0] | float
    elevation: AtIndex[1] | float
    fit_error: MetaData[Optional[float]] = None


@dataclass
class RangeMeasurement(NamedArray):
    range: AtIndex[0] | float
    time_of_flight: MetaData[Optional[float]] = None
    accuracy: MetaData[Optional[float]] = None


@dataclass
class DepthMeasurement(NamedArray):
    depth: AtIndex[0] | float
    accuracy: MetaData[Optional[float]] = None