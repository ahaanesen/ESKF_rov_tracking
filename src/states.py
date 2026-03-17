import numpy as np
from dataclasses import dataclass
from typing import Optional

from quaternion import RotationQuaterion
from senfuslib import NamedArray, AtIndex, MetaData, MultiVarGauss

# -----------------------------------------------------------------------------
# Common helper
# -----------------------------------------------------------------------------

@dataclass
class WithXYZ(NamedArray):
    x: AtIndex[0]
    y: AtIndex[1]
    z: AtIndex[2]
    xy: AtIndex[0:2]


# -----------------------------------------------------------------------------
# ASV state (15-state ESKF nominal + 15-state error)
# -----------------------------------------------------------------------------

@dataclass
class ASVNominalState(NamedArray):
    """
    ASV nominal ESKF state in NED.
    16 elements (pos(3), vel(3), quat(4), acc_bias(3), gyro_bias(3))
    """
    pos: AtIndex[0:3] | WithXYZ
    vel: AtIndex[3:6] | WithXYZ
    ori: AtIndex[6:10] | RotationQuaterion
    accm_bias: AtIndex[10:13] | WithXYZ
    gyro_bias: AtIndex[13:16] | WithXYZ

    def diff(self, other: 'ASVNominalState') -> 'ASVErrorState':
        """Calculate the difference between two nominal states.
        Used to calculate NEES.
        Returns:
            ASVErrorState: error state representing the difference
        """
        return ASVErrorState(
            pos=self.pos - other.pos,
            vel=self.vel - other.vel,
            avec=self.ori.diff_as_avec(other.ori),
            accm_bias=self.accm_bias - other.accm_bias,
            gyro_bias=self.gyro_bias - other.gyro_bias)
    
    @property
    def euler(self) -> np.ndarray:
        """Orientation as euler angles (roll, pitch, yaw) in NED"""
        return WithXYZ.from_array(self.ori.as_euler())


@dataclass
class ASVErrorState(NamedArray):
    """
    ASV error-state (15): dp(3), dv(3), dtheta(3), dab(3), dgb(3)
    """
    pos: AtIndex[0:3] | WithXYZ
    vel: AtIndex[3:6] | WithXYZ
    avec: AtIndex[6:9] | WithXYZ
    accm_bias: AtIndex[9:12] | WithXYZ
    gyro_bias: AtIndex[12:15] | WithXYZ


# -----------------------------------------------------------------------------
# ROV CV state (pos+vel only)
# -----------------------------------------------------------------------------

@dataclass
class ROVNominalCV(NamedArray):
    """ROV nominal CV state in NED: pos(3), vel(3)"""
    pos: AtIndex[0:3] | WithXYZ
    vel: AtIndex[3:6] | WithXYZ


@dataclass
class ROVErrorCV(NamedArray):
    """ROV CV error-state: dp(3), dv(3)"""
    pos: AtIndex[0:3] | WithXYZ
    vel: AtIndex[3:6] | WithXYZ


# -----------------------------------------------------------------------------
# Joint state (ASV + ROV)
# -----------------------------------------------------------------------------

class JointIdx:
    """
    Central place for augmented error-state indexing (length 21):
      ASV: 0..14
      ROV: 15..20
    """
    N_ASV = 15
    N_ROV = 6
    N = N_ASV + N_ROV

    # ASV slices
    ASV = slice(0, 15)
    ASV_POS = slice(0, 3)
    ASV_VEL = slice(3, 6)
    ASV_AVEC = slice(6, 9)
    ASV_AB = slice(9, 12)
    ASV_GB = slice(12, 15)

    # ROV slices (offset by 15)
    ROV = slice(15, 21)
    ROV_POS = slice(15, 18)
    ROV_VEL = slice(18, 21)


@dataclass
class JointNominalState:
    """
    Not a NamedArray on purpose: it’s two different nominal types
    with different sizes (16 vs 6). Keeping them separate makes
    prediction/injection much simpler and avoids fake quaternion for ROV.
    """
    asv: ASVNominalState
    rov: ROVNominalCV


@dataclass
class JointErrorState(NamedArray):
    """
    Flattened joint error mean (21) stored as a NamedArray for convenience:
      [ASV(15), ROV(6)]
    """
    # ASV
    asv_pos: AtIndex[0:3] | WithXYZ
    asv_vel: AtIndex[3:6] | WithXYZ
    asv_avec: AtIndex[6:9] | WithXYZ
    asv_accm_bias: AtIndex[9:12] | WithXYZ
    asv_gyro_bias: AtIndex[12:15] | WithXYZ
    # ROV (CV)
    rov_pos: AtIndex[15:18] | WithXYZ
    rov_vel: AtIndex[18:21] | WithXYZ


@dataclass
class JointEskfState:
    nom: JointNominalState
    err: MultiVarGauss[JointErrorState]  # mean is JointErrorState, cov is (21x21)

    def get_err_gauss(self, gt: JointNominalState) -> MultiVarGauss[JointErrorState]:
        """Calculate the error Gaussian given a ground truth nominal state."""
        err = JointErrorState(
            asv_pos=self.nom.asv.pos - gt.asv.pos,
            asv_vel=self.nom.asv.vel - gt.asv.vel,
            asv_avec=gt.asv.ori.diff_as_avec(self.nom.asv.ori),
            asv_accm_bias=self.nom.asv.accm_bias - gt.asv.accm_bias,
            asv_gyro_bias=self.nom.asv.gyro_bias - gt.asv.gyro_bias,
            rov_pos=self.nom.rov.pos - gt.rov.pos,
            rov_vel=self.nom.rov.vel - gt.rov.vel)
        return MultiVarGauss[JointErrorState](err, self.err.cov)


def zero_joint_error() -> JointErrorState:
    return JointErrorState.from_array(np.zeros(JointIdx.N))