from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import scipy.linalg

from senfuslib import MultiVarGauss

from quaternion import RotationQuaterion
from utils.indexing import block_3x3
from utils.cross_matrix import get_cross_matrix

from tracking_and_navigation.states import (AsvNominalState,
                    RovNominalCV, 
                    RovErrorCV)

from tracking_and_navigation.measurements import ImuMeasurement, CorrectedImuMeasurement


# =============================================================================
# IMU MODEL (unchanged): 15-state ESKF dynamics
# =============================================================================

@dataclass
class ModelIMU:
    """IMU-driven 15-state ESKF dynamic model (used for ASV)."""

    accm_std: float
    accm_bias_std: float
    accm_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    accm_correction: "np.ndarray"  # (3,3)
    gyro_correction: "np.ndarray"  # (3,3)

    # g: "np.ndarray" = field(default=np.array([0, 0, 9.82]))
    g: np.ndarray = field(default_factory=lambda: np.array([0, 0, 9.81]))

    Q_c: "np.ndarray" = field(init=False, repr=False)

    def __post_init__(self):
        def diag3(x):
            return np.diag([x] * 3)

        accm_corr = self.accm_correction
        gyro_corr = self.gyro_correction

        self.Q_c = scipy.linalg.block_diag(
            accm_corr @ diag3(self.accm_std**2) @ accm_corr.T,
            gyro_corr @ diag3(self.gyro_std**2) @ gyro_corr.T,
            diag3(self.accm_bias_std**2),
            diag3(self.gyro_bias_std**2),
        )

    def correct_z_imu(self, x_est_nom: AsvNominalState, z_imu: ImuMeasurement) -> CorrectedImuMeasurement:
        acc_est = self.accm_correction @ (z_imu.acc - x_est_nom.accm_bias)
        avel_est = self.gyro_correction @ (z_imu.avel - x_est_nom.gyro_bias)
        return CorrectedImuMeasurement(acc_est, avel_est)

    def predict_nom(self, x_est_nom: AsvNominalState, z_corr: CorrectedImuMeasurement, dt: float) -> AsvNominalState:
        Rq = x_est_nom.ori.as_rotmat()
        acc_world = Rq @ z_corr.acc + self.g

        pos_pred = x_est_nom.pos + dt * x_est_nom.vel + (dt**2) / 2 * acc_world
        vel_pred = x_est_nom.vel + dt * acc_world

        delta_rot = RotationQuaterion.from_avec(z_corr.avel * dt)
        ori_pred = x_est_nom.ori @ delta_rot

        acc_bias_pred = x_est_nom.accm_bias
        gyro_bias_pred = x_est_nom.gyro_bias

        return AsvNominalState(pos_pred, vel_pred, ori_pred, acc_bias_pred, gyro_bias_pred)

    def A_c(self, x_est_nom: AsvNominalState, z_corr: CorrectedImuMeasurement) -> np.ndarray:
        A_c = np.zeros((15, 15))
        Rq = x_est_nom.ori.as_rotmat()
        S_acc = get_cross_matrix(z_corr.acc)
        S_omega = get_cross_matrix(z_corr.avel)

        A_c[block_3x3(0, 1)] = np.eye(3)
        A_c[block_3x3(1, 2)] = -Rq @ S_acc
        A_c[block_3x3(1, 3)] = -Rq @ self.accm_correction
        A_c[block_3x3(2, 2)] = -S_omega
        A_c[block_3x3(2, 4)] = -self.gyro_correction

        return A_c

    def get_error_G_c(self, x_est_nom: AsvNominalState) -> np.ndarray:
        G_c = np.zeros((15, 12))
        Rq = x_est_nom.ori.as_rotmat()

        G_c[block_3x3(1, 0)] = -Rq
        G_c[block_3x3(2, 1)] = -np.eye(3)
        G_c[block_3x3(3, 2)] = np.eye(3)
        G_c[block_3x3(4, 3)] = np.eye(3)

        return G_c

    def get_discrete_error_diff(
        self,
        x_est_nom: AsvNominalState,
        z_corr: CorrectedImuMeasurement,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        A_c = self.A_c(x_est_nom, z_corr)
        G_c = self.get_error_G_c(x_est_nom)
        GQGT_c = G_c @ self.Q_c @ G_c.T

        n = A_c.shape[0]
        exponent = np.zeros((2 * n, 2 * n))
        exponent[:n, :n] = -A_c
        exponent[:n, n:] = GQGT_c
        exponent[n:, n:] = A_c.T
        exponent *= dt

        VanLoanMatrix = scipy.linalg.expm(exponent)
        V1 = VanLoanMatrix[n:, n:]
        V2 = VanLoanMatrix[:n, n:]
        A_d = V1.T
        GQGT_d = V1.T @ V2

        return A_d, GQGT_d

    # # Note: keeping predict_err for legacy code paths.
    # # Joint ESKF should NOT use this; it should use get_discrete_error_diff().
    # def predict_err(self, x_est_prev: EskfState, z_corr: CorrectedImuMeasurement, dt: float) -> MultiVarGauss[ErrorState]:
    #     x_est_prev_nom = x_est_prev.nom
    #     Ad, GQGTd = self.get_discrete_error_diff(x_est_prev_nom, z_corr, dt)

    #     P_prev = x_est_prev.err.cov
    #     P_pred = Ad @ P_prev @ Ad.T + GQGTd

    #     mean_pred = np.zeros(15)
    #     return MultiVarGauss[ErrorState](ErrorState.from_array(mean_pred), P_pred)



# =============================================================================
# CV MODEL (NEW): 6-state constant velocity for ROV
# =============================================================================

@dataclass
class ModelCV:
    """
    6-state constant velocity model for ROV:
      nominal: pos(3), vel(3)
      error:   dpos(3), dvel(3)

    sigma_a is the (1-sigma) acceleration noise [m/s^2] driving the model.
    """
    sigma_a: float

    def predict_nom(self, rov_nom: RovNominalCV, dt: float) -> RovNominalCV:
        pos_pred = rov_nom.pos + dt * rov_nom.vel
        vel_pred = rov_nom.vel
        return RovNominalCV(pos_pred, vel_pred)

    def F_Q(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Discrete-time error dynamics for 6-state CV:
          x = [dp, dv]
          F = [[I, dtI],
               [0,  I]]
          Q from continuous white accel noise sigma_a.
        """
        I = np.eye(3)
        Z = np.zeros((3, 3))

        F = np.block([[I, dt * I], [Z, I]])  # (6,6)

        Q_pp = (dt**4) / 4.0 * (self.sigma_a**2) * I
        Q_pv = (dt**3) / 2.0 * (self.sigma_a**2) * I
        Q_vv = (dt**2) * (self.sigma_a**2) * I

        Q = np.block([[Q_pp, Q_pv], [Q_pv, Q_vv]])  # (6,6)
        return F, Q

    def predict_err(self, err_prev: MultiVarGauss[RovErrorCV], dt: float) -> MultiVarGauss[RovErrorCV]:
        """
        Predict only the ROV CV error Gaussian (6x6).
        Mean remains zero.
        """
        P_prev = err_prev.cov
        F, Q = self.F_Q(dt)
        P_pred = F @ P_prev @ F.T + Q
        return MultiVarGauss(RovErrorCV.from_array(np.zeros(6)), P_pred)