import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Union

# if TYPE_CHECKING:  # used to avoid circular imports with solution
from quaternion import RotationQuaterion
from senfuslib import NamedArray, AtIndex, MetaData
from senfuslib import MultiVarGauss

from config import DEBUG

@dataclass
class WithXYZ(NamedArray):
    x: AtIndex[0]
    y: AtIndex[1]
    z: AtIndex[2]
    xy: AtIndex[0:2]


@dataclass
class ASVState(NamedArray):
    """
    ASV pose provided at each USBL measurement epoch.
    Both fields are considered ground-truth (RTK quality).

    pos (ndarray[3]): position in NED
    ori (RotationQuaterion): orientation as a quaternion in NED
    """
    pos: AtIndex[0:3] | WithXYZ
    ori: AtIndex[3:7] | RotationQuaterion



@ dataclass
class UsblMeasurement(NamedArray):
    """Represents data received from a USBL sensor, consists of azimuth and elevation measurements.
    Args:
        azimuth (float): the azimuth measurement. Values between [0, 2pi], where 0 is north, and increases clockwise.
        elevation (float): the elevation measurement. Values between [-pi/2, pi/2], where 0 is horizontal, positive is up, and negative is down.
        fit_error (float): the fit error value indicates the quality of fit (or confidence) of the azimuth and elevation values. 
                Lower values (0.0) indicate better fit, while larger values (<2.0-3.0) indicate poorer fits.
    """
    azimuth: AtIndex[0] | float
    elevation: AtIndex[1] | float
    fit_error: MetaData[Optional[float]] = None

@ dataclass
class RangeMeasurement(NamedArray):
    """Represents a psudo-range measurement calculated through one-way travel time of acoustic messages.
    Args:
        range (float): the range measurement in meters.
        accuracy (Optional[float]): the reported accuracy from the sensor (not used).
    """
    range: AtIndex[0] | float
    time_of_flight: MetaData[Optional[float]] = None
    accuracy: MetaData[Optional[float]] = None
