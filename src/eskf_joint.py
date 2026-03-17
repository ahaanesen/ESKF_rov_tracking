import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from senfuslib import MultiVarGauss

from models import ModelIMU, ModelCV
from quaternion import RotationQuaterion
from utils.cross_matrix import get_cross_matrix
from utils.angles import wrap_to_pi

from states import (
    JointEskfState,
    JointNominalState,
    JointErrorState,
    JointIdx,
    ASVNominalState,
    ROVNominalCV,
)

from measurements import (
    ImuMeasurement,
    CorrectedImuMeasurement,
    GnssMeasurement,
    UsblMeasurement,
    RangeMeasurement,
    DepthMeasurement,
)

from sensors import (
    SensorGNSS_ASV,
    SensorUSBL_Joint,
    SensorRange_Joint,
    SensorDepth_ROV,
)


def _as_measurement_array(z: Any) -> np.ndarray:
    return np.asarray(z, dtype=float).reshape(-1)


class _ESKFJointShared:
    @staticmethod
    def _require_sensor(sensor: Any, sensor_name: str) -> Any:
        if sensor is None:
            raise ValueError(f"{sensor_name} is not configured for this ESKF instance")
        return sensor

    def _update_err(
        self,
        x_pred: JointEskfState,
        z_pred_gauss: MultiVarGauss,
        z_meas: Any,
        H: np.ndarray,
        R: np.ndarray,
        wrap_first_angle: bool = False,
    ) -> MultiVarGauss[JointErrorState]:
        """
        Generic Kalman measurement update on joint error-state.

        x_pred.err.cov is (21,21)
        H is (m,21)
        R is (m,m)
        z_pred_gauss is MultiVarGauss[z](z_pred, S)
        """
        P = x_pred.err.cov
        z_pred, S = z_pred_gauss

        z_pred_arr = _as_measurement_array(z_pred)
        z_meas_arr = _as_measurement_array(z_meas)
        innovation = z_meas_arr - z_pred_arr

        if wrap_first_angle and innovation.size > 0:
            innovation[0] = wrap_to_pi(innovation[0])

        # W = P H^T S^{-1} (computed as solve)
        W = np.linalg.solve(S.T, H @ P.T).T  # (21,m)
        dx = W @ innovation  # (21,)

        # Joseph form covariance update for stability
        I_WH = np.eye(P.shape[0]) - W @ H
        P_upd = I_WH @ P @ I_WH.T + W @ R @ W.T

        return MultiVarGauss(JointErrorState.from_array(dx), P_upd)

    @staticmethod
    def _inject(
        x_nom: JointNominalState,
        dx_gauss: MultiVarGauss[JointErrorState],
    ) -> JointEskfState:
        """
        Inject joint error into joint nominal, then reset mean to zero and
        transform covariance with joint injection Jacobian G.

        ASV injection:
          p, v, biases additive
          q <- q ⊗ [1, 0.5*dtheta]
          G_asv[6:9,6:9] = I - [0.5*dtheta]_x

        ROV (CV) injection:
          p, v additive
          G_rov = I6
        """
        dx = dx_gauss.mean
        P = dx_gauss.cov

        # --- ASV part (15)
        d_asv_pos = np.asarray(dx.asv_pos, dtype=float).reshape(3)
        d_asv_vel = np.asarray(dx.asv_vel, dtype=float).reshape(3)
        d_asv_avec = np.asarray(dx.asv_avec, dtype=float).reshape(3)
        d_asv_ab = np.asarray(dx.asv_accm_bias, dtype=float).reshape(3)
        d_asv_gb = np.asarray(dx.asv_gyro_bias, dtype=float).reshape(3)

        asv_nom = x_nom.asv
        asv_pos_inj = asv_nom.pos + d_asv_pos
        asv_vel_inj = asv_nom.vel + d_asv_vel
        asv_ori_inj = asv_nom.ori.multiply(RotationQuaterion(1, 0.5 * d_asv_avec))
        asv_ab_inj = asv_nom.accm_bias + d_asv_ab
        asv_gb_inj = asv_nom.gyro_bias + d_asv_gb

        asv_inj = ASVNominalState(asv_pos_inj, asv_vel_inj, asv_ori_inj, asv_ab_inj, asv_gb_inj)

        # --- ROV part (6)
        d_rov_pos = np.asarray(dx.rov_pos, dtype=float).reshape(3)
        d_rov_vel = np.asarray(dx.rov_vel, dtype=float).reshape(3)

        rov_nom = x_nom.rov
        rov_pos_inj = rov_nom.pos + d_rov_pos
        rov_vel_inj = rov_nom.vel + d_rov_vel
        rov_inj = ROVNominalCV(rov_pos_inj, rov_vel_inj)

        nom_inj = JointNominalState(asv=asv_inj, rov=rov_inj)

        # --- Joint injection Jacobian G (21x21)
        G = np.eye(JointIdx.N)

        # ASV attitude injection effect on covariance
        G_asv = np.eye(15)
        G_asv[6:9, 6:9] -= get_cross_matrix(0.5 * d_asv_avec)
        G[JointIdx.ASV, JointIdx.ASV] = G_asv

        # ROV is identity (already)
        P_inj = G @ P @ G.T

        err_inj = MultiVarGauss(JointErrorState.from_array(np.zeros(JointIdx.N)), P_inj)
        return JointEskfState(nom=nom_inj, err=err_inj)


@dataclass
class ESKF_joint(_ESKFJointShared):
    """
    Joint ESKF:
      - ASV: IMU-driven ESKF (15 error states) via ModelIMU
      - ROV: CV (6 error states)
    """
    modelImuAsv: ModelIMU
    modelCvRov: ModelCV

    # Optional sensors (configure as needed)
    sensorGnssAsv: Optional[SensorGNSS_ASV] = None
    sensorUsbl: Optional[SensorUSBL_Joint] = None
    sensorRange: Optional[SensorRange_Joint] = None
    sensorDepth: Optional[SensorDepth_ROV] = None

    def predict_from_imu(
        self,
        x_prev: JointEskfState,
        z_imu_asv: ImuMeasurement,
        dt: float,
    ) -> JointEskfState:
        """
        Predict step driven by ASV IMU.
        Propagates:
          - ASV nominal with ModelIMU
          - ROV nominal with CV
          - joint covariance with F = blockdiag(Ad_asv, F_rov) and Q blockdiag(...)
        """
        if dt == 0.0:
            return x_prev

        # ---- 1) ASV nominal propagation (reuse your ModelIMU)
        asv_prev = x_prev.nom.asv
        z_corr = self.modelImuAsv.correct_z_imu(asv_prev, z_imu_asv)
        asv_pred = self.modelImuAsv.predict_nom(asv_prev, z_corr, dt)

        # ---- 2) ROV nominal CV propagation
        rov_prev = x_prev.nom.rov
        rov_pred = self.modelCvRov.predict_nom(rov_prev, dt)

        nom_pred = JointNominalState(asv=asv_pred, rov=rov_pred)

        # ---- 3) Joint covariance propagation (21x21)
        P_prev = x_prev.err.cov

        Ad_asv, GQGTd_asv = self.modelImuAsv.get_discrete_error_diff(asv_prev, z_corr, dt)  # (15,15)
        F_rov, Q_rov = self.modelCvRov.F_Q(dt)  # (6,6)

        F = np.eye(JointIdx.N)
        Q = np.zeros((JointIdx.N, JointIdx.N))
        F[JointIdx.ASV, JointIdx.ASV] = Ad_asv
        F[JointIdx.ROV, JointIdx.ROV] = F_rov

        Q[JointIdx.ASV, JointIdx.ASV] = GQGTd_asv
        Q[JointIdx.ROV, JointIdx.ROV] = Q_rov

        P_pred = F @ P_prev @ F.T + Q

        err_pred = MultiVarGauss(JointErrorState.from_array(np.zeros(JointIdx.N)), P_pred)
        return JointEskfState(nom=nom_pred, err=err_pred)

    # ---- Example updates (optional, but shows how to use _update_err + _inject)

    def update_from_gnss_asv(
        self,
        x_pred: JointEskfState,
        z_gnss: GnssMeasurement,
    ) -> Tuple[JointEskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorGnssAsv, "sensorGnssAsv")
        z_pred = sensor.pred_from_est(x_pred)
        H = sensor.H(x_pred.nom.asv)  # (3,21)
        R = sensor.R
        dx = self._update_err(x_pred, z_pred, z_gnss, H, R, wrap_first_angle=False)
        return self._inject(x_pred.nom, dx), z_pred

    def update_from_usbl(
        self,
        x_pred: JointEskfState,
        z_usbl: UsblMeasurement,
    ) -> Tuple[JointEskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorUsbl, "sensorUsbl")
        z_pred = sensor.pred_from_est(x_pred)
        H = sensor.H(x_pred.nom.asv, x_pred.nom.rov)  # (2,21)
        R = sensor.R
        dx = self._update_err(x_pred, z_pred, z_usbl, H, R, wrap_first_angle=True)
        return self._inject(x_pred.nom, dx), z_pred

    def update_from_range(
        self,
        x_pred: JointEskfState,
        z_range: RangeMeasurement,
    ) -> Tuple[JointEskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorRange, "sensorRange")
        z_pred = sensor.pred_from_est(x_pred)
        H = sensor.H(x_pred.nom.asv, x_pred.nom.rov)  # (1,21)
        R = sensor.R
        dx = self._update_err(x_pred, z_pred, z_range, H, R, wrap_first_angle=False)
        return self._inject(x_pred.nom, dx), z_pred

    def update_from_depth(
        self,
        x_pred: JointEskfState,
        z_depth: DepthMeasurement,
    ) -> Tuple[JointEskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorDepth, "sensorDepth")
        z_pred = sensor.pred_from_est(x_pred)
        H = sensor.H()  # (1,21) in the sensor design I suggested
        R = sensor.R
        dx = self._update_err(x_pred, z_pred, z_depth, H, R, wrap_first_angle=False)
        return self._inject(x_pred.nom, dx), z_pred