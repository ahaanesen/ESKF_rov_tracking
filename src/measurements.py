from dataclasses import dataclass
from typing import Optional

from senfuslib import NamedArray, AtIndex, MetaData
from states import WithXYZ


# -------------------------
# ASV navigation measurements
# -------------------------

@dataclass
class ImuMeasurement(NamedArray):
    acc: AtIndex[0:3] | WithXYZ
    avel: AtIndex[3:6] | WithXYZ

@dataclass
class CorrectedImuMeasurement(ImuMeasurement):
    """Represents processed data from the IMU.
    Corrected for axis alignmentand scale scale, and bias. 
    Not 'corrected' for gravity.
    """

@dataclass
class GnssMeasurement(NamedArray):
    pos: AtIndex[0:3] | WithXYZ
    accuracy: MetaData[Optional[float]] = None


# -------------------------
# Tracking / coupling measurements
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