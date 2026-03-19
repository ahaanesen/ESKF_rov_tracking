import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Union

# if TYPE_CHECKING:  # used to avoid circular imports with solution
from quaternion import RotationQuaterion
from senfuslib import NamedArray, AtIndex, MetaData
from senfuslib import MultiVarGauss

@dataclass
class WithXYZ(NamedArray):
    x: AtIndex[0]
    y: AtIndex[1]
    z: AtIndex[2]
    xy: AtIndex[0:2]

# -----------------------------------------------------------------------------
# ASV state (not to be estimated), considered ground-truth (RTK quality)
# -----------------------------------------------------------------------------
@dataclass
class ASVState(NamedArray):
    """
    ASV pose.
    Both fields are considered ground-truth (RTK quality).

    pos (ndarray[3]): position in NED
    ori (RotationQuaterion): orientation as a quaternion in NED
    """
    pos: AtIndex[0:3] | WithXYZ
    ori: AtIndex[3:7] | RotationQuaterion

# -----------------------------------------------------------------------------
# ROV state (to be estimated)
# -----------------------------------------------------------------------------
# TODO: Is still IMU based and should be CV based
@dataclass
class RovNominalState(NamedArray):
    """Class representing a nominal state. See (Table 10.1) in the book.

    Args:
        pos (ndarray[3]): position in NED
        vel (ndarray[3]): velocity in NED
        ori (RotationQuaterion): orientation as a quaternion in NED
        accm_bias (ndarray[3]): accelerometer bias
        gyro_bias (ndarray[3]): gyro bias
    """
    pos: AtIndex[0:3] | WithXYZ
    vel: AtIndex[3:6] | WithXYZ
    ori: AtIndex[6:10] | RotationQuaterion
    accm_bias: AtIndex[10:13] | WithXYZ
    gyro_bias: AtIndex[13:16] | WithXYZ

    def diff(self, other: 'RovNominalState') -> 'RovErrorState':
        """Calculate the difference between two nominal states.
        Used to calculate NEES.
        Returns:
            ErrorState: error state representing the difference
        """
        return RovNominalState(
            pos=self.pos - other.pos,
            vel=self.vel - other.vel,
            ori=self.ori.diff_as_avec(other.ori),
            accm_bias=self.accm_bias - other.accm_bias,
            gyro_bias=self.gyro_bias - other.gyro_bias)

    @property
    def euler(self) -> np.ndarray:
        """Orientation as euler angles (roll, pitch, yaw) in NED"""
        return WithXYZ.from_array(self.ori.as_euler())


@dataclass
class RovErrorState(NamedArray):
    """Class representing a nominal state. See (Table 10.1) in the book."""
    pos: AtIndex[0:3] | WithXYZ
    vel: AtIndex[3:6] | WithXYZ
    avec: AtIndex[6:9] | WithXYZ
    accm_bias: AtIndex[9:12] | WithXYZ
    gyro_bias: AtIndex[12:15] | WithXYZ


@dataclass
class RovEskfState:
    """A combination of nominal and error state"""
    nom: RovNominalState
    err: MultiVarGauss[RovErrorState]

    def get_err_gauss(self, gt: RovNominalState) -> MultiVarGauss[RovErrorState]:
        """Used to calculate error and NEES"""
        err = RovErrorState(
            pos=self.nom.pos - gt.pos,
            vel=self.nom.vel - gt.vel,
            avec=gt.ori.diff_as_avec(self.nom.ori),
            accm_bias=self.nom.accm_bias - gt.accm_bias,
            gyro_bias=self.nom.gyro_bias - gt.gyro_bias)
        return MultiVarGauss[RovErrorState](err, self.err.cov)
