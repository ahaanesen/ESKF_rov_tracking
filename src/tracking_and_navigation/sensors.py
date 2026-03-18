from dataclasses import dataclass, field
import numpy as np

from senfuslib import MultiVarGauss
from quaternion import RotationQuaterion

from tracking_and_navigation.states import (
    JointEskfState,
    JointIdx,
    ASVNominalState,
    ROVNominalCV,
)
from tracking_and_navigation.measurements import GnssMeasurement, UsblMeasurement, RangeMeasurement, DepthMeasurement
from utils.cross_matrix import get_cross_matrix
from utils.angles import wrap_to_2pi, wrap_to_pi


# -----------------------------------------------------------------------------
# ASV GNSS sensor (navigation)
# -----------------------------------------------------------------------------

@dataclass
class SensorGNSS_ASV:
    gnss_std_ne: float
    gnss_std_d: float
    lever_arm: "np.ndarray"  # (3,)
    R: "np.ndarray" = field(init=False)

    def __post_init__(self):
        self.R = np.diag([self.gnss_std_ne**2, self.gnss_std_ne**2, self.gnss_std_d**2])

    def H(self, asv_nom: ASVNominalState) -> np.ndarray:
        """
        Return H of shape (3, 21). Only ASV part is filled.
        Same as your old GNSS H, just embedded in the joint error state.
        """
        H = np.zeros((3, JointIdx.N))
        # dpos_meas / d(asv_pos_err) = I
        H[:, JointIdx.ASV_POS] = np.eye(3)
        # lever arm sensitivity wrt attitude error (dtheta)
        H[:, JointIdx.ASV_AVEC] = -asv_nom.ori.as_rotmat() @ get_cross_matrix(self.lever_arm)
        return H

    def pred_from_est(self, x_est: JointEskfState) -> MultiVarGauss[GnssMeasurement]:
        asv = x_est.nom.asv
        P = x_est.err.cov
        z_pred = asv.pos + asv.ori.as_rotmat() @ self.lever_arm
        H = self.H(asv)
        S = self.R + H @ P @ H.T
        return MultiVarGauss(GnssMeasurement.from_array(z_pred), S)


# -----------------------------------------------------------------------------
# Joint USBL sensor (ASV pose + ROV position)
# -----------------------------------------------------------------------------

@dataclass
class SensorUSBL_Joint:
    usbl_std: float
    lever_arm_asv: "np.ndarray"  # (3,) lever arm from ASV origin to USBL head, in ASV body
    R: "np.ndarray" = field(init=False)

    def __post_init__(self):
        self.R = np.diag([self.usbl_std**2, self.usbl_std**2])

    def _relative_vector_ned(self, asv: ASVNominalState, rov: ROVNominalCV) -> np.ndarray:
        p_usbl = asv.pos + asv.ori.as_rotmat() @ self.lever_arm_asv
        return rov.pos - p_usbl  # d = p_rov - p_usbl

    def pred_from_est(self, x_est: JointEskfState) -> MultiVarGauss[UsblMeasurement]:
        asv = x_est.nom.asv
        rov = x_est.nom.rov
        P = x_est.err.cov

        d = self._relative_vector_ned(asv, rov)
        dx, dy, dz = d
        r = np.sqrt(dx**2 + dy**2)

        az = wrap_to_2pi(np.arctan2(dy, dx))
        el = np.arctan2(dz, r)

        z_pred = np.array([az, el])
        H = self.H(asv, rov)
        S = self.R + H @ P @ H.T

        return MultiVarGauss(UsblMeasurement.from_array(z_pred), S)

    def H(self, asv: ASVNominalState, rov: ROVNominalCV) -> np.ndarray:
        """
        Joint Jacobian H: shape (2, 21)

        Measurement depends on d = p_rov - (p_asv + R_asv*lever_arm).
        So:
          ∂h/∂p_rov != 0   (ROV pos block)
          ∂h/∂p_asv != 0   (ASV pos block)
          ∂h/∂theta_asv != 0 (ASV attitude error block, via R_asv*lever arm)
        """
        H = np.zeros((2, JointIdx.N))

        d = self._relative_vector_ned(asv, rov)
        dx, dy, dz = d

        r_sq = dx**2 + dy**2
        r = np.sqrt(r_sq)
        rho_sq = r_sq + dz**2

        if r < 1e-6 or rho_sq < 1e-9:
            return H

        # dh/dd (same as you already had)
        dh_dd = np.array([
            [-dy/r_sq,               dx/r_sq,               0.0],
            [-(dx*dz)/(rho_sq*r),    -(dy*dz)/(rho_sq*r),   r/rho_sq],
        ])  # (2,3)

        # d depends on rov.pos and asv.pos linearly:
        # d = rov.pos - asv.pos - R_asv*lever
        # ∂d/∂rov_pos_err = +I
        # ∂d/∂asv_pos_err = -I
        H[:, JointIdx.ROV_POS] = dh_dd @ np.eye(3)
        H[:, JointIdx.ASV_POS] = dh_dd @ (-np.eye(3))

        # attitude: variation of (R*lever) wrt small angle error dtheta:
        # δ(R*lever) ≈ -R * [lever]_x * δtheta   (depending on your convention)
        # Since d = ... - R*lever, we get:
        # δd_att ≈ - (δ(R*lever)) ≈ + R*[lever]_x*δtheta
        R_asv = asv.ori.as_rotmat()
        dd_dtheta = R_asv @ get_cross_matrix(self.lever_arm_asv)  # (3,3)

        H[:, JointIdx.ASV_AVEC] = dh_dd @ dd_dtheta

        return H


# -----------------------------------------------------------------------------
# Joint Range sensor (ASV pose + ROV position)
# -----------------------------------------------------------------------------

@dataclass
class SensorRange_Joint:
    range_std: float
    lever_arm_asv: "np.ndarray"
    R: "np.ndarray" = field(init=False)

    def __post_init__(self):
        self.R = np.array([[self.range_std**2]])

    def pred_from_est(self, x_est: JointEskfState) -> MultiVarGauss[RangeMeasurement]:
        asv = x_est.nom.asv
        rov = x_est.nom.rov
        P = x_est.err.cov

        p_sensor = asv.pos + asv.ori.as_rotmat() @ self.lever_arm_asv
        d = rov.pos - p_sensor
        rho = float(np.linalg.norm(d))

        z_pred = np.array([rho])
        H = self.H(asv, rov)
        S = self.R + H @ P @ H.T
        return MultiVarGauss(RangeMeasurement.from_array(z_pred), S)

    def H(self, asv: ASVNominalState, rov: ROVNominalCV) -> np.ndarray:
        H = np.zeros((1, JointIdx.N))
        
        R_asv = asv.ori.as_rotmat()
        p_sensor = asv.pos + R_asv @ self.lever_arm_asv
        d = rov.pos - p_sensor
        rho = float(np.linalg.norm(d))
        if rho < 1e-6:
            return H

        dr_dd = (d / rho).reshape(1, 3)  # (1,3)

        # ∂d/∂rov_pos = +I, ∂d/∂asv_pos = -I
        H[:, JointIdx.ROV_POS] = dr_dd
        H[:, JointIdx.ASV_POS] = -dr_dd

        # attitude term: same logic as USBL
        dd_dtheta = R_asv @ get_cross_matrix(self.lever_arm_asv)  # (3,3)
        H[:, JointIdx.ASV_AVEC] = dr_dd @ dd_dtheta

        return H


# -----------------------------------------------------------------------------
# ROV Depth sensor (ROV only)
# -----------------------------------------------------------------------------

@dataclass
class SensorDepth_ROV:
    depth_std: float
    R: "np.ndarray" = field(init=False)

    def __post_init__(self):
        self.R = np.array([[self.depth_std**2]])

    def pred_from_est(self, x_est: JointEskfState) -> MultiVarGauss[DepthMeasurement]:
        rov = x_est.nom.rov
        P = x_est.err.cov
        z_pred = np.array([rov.pos[2]])  # NED: z is "down" in your convention
        H = self.H()
        S = self.R + H @ P @ H.T
        return MultiVarGauss(DepthMeasurement.from_array(z_pred), S)

    def H(self) -> np.ndarray:
        H = np.zeros((1, JointIdx.N))
        # ROV pos z is index 15+2
        H[0, JointIdx.ROV_POS.start + 2] = 1.0
        return H