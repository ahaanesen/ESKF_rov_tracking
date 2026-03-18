import numpy as np

import scipy.linalg
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from senfuslib import MultiVarGauss
from tracking_only.rov_states import NominalState, ErrorState, EskfState, ImuMeasurement, DepthMeasurement
from tracking_only.asv_states import ASVState, UsblMeasurement, RangeMeasurement

from tracking_only.rov_sensors import SensorDepth
from tracking_only.asv_sensors import SensorUSBL, SensorRange

from quaternion import RotationQuaterion
from utils.cross_matrix import get_cross_matrix
from utils.angles import wrap_to_pi
from tracking_only.models import ModelIMU, ModelCV


class _ESKFUpdateShared:
    @staticmethod
    def _as_measurement_array(z_measurement: Any) -> np.ndarray:
        return np.asarray(z_measurement, dtype=float).reshape(-1)

    @staticmethod
    def _require_sensor(sensor: Any, sensor_name: str) -> Any:
        if sensor is None:
            raise ValueError(f"{sensor_name} is not configured for this ESKF instance")
        return sensor

    def _update_err(
        self,
        rov_est_pred: EskfState,
        z_pred_gauss: MultiVarGauss,
        z_measurement: Any,
        H: np.ndarray,
        R: np.ndarray,
        is_usbl: bool = False,
    ) -> MultiVarGauss[ErrorState]:
        P = rov_est_pred.err.cov
        z_pred, S = z_pred_gauss

        z_pred_arr = self._as_measurement_array(z_pred)
        z_meas_arr = self._as_measurement_array(z_measurement)
        innovation = z_meas_arr - z_pred_arr

        if is_usbl:
            innovation[0] = wrap_to_pi(innovation[0])

        W = np.linalg.solve(S.T, H @ P.T).T
        rov_err_upd_mean = W @ innovation

        I_WH = np.eye(P.shape[0]) - W @ H
        rov_err_cov_upd = I_WH @ P @ I_WH.T + W @ R @ W.T

        return MultiVarGauss[ErrorState](
            ErrorState.from_array(rov_err_upd_mean), rov_err_cov_upd
        )

    @staticmethod
    def _inject(
        rov_est_nom: NominalState,
        rov_est_err: MultiVarGauss[ErrorState],
    ) -> EskfState:
        pos_inj = rov_est_nom.pos + rov_est_err.mean.pos
        vel_inj = rov_est_nom.vel + rov_est_err.mean.vel
        ori_inj = rov_est_nom.ori.multiply(
            RotationQuaterion(1, 0.5 * rov_est_err.mean.avec)
        )
        accm_bias_inj = rov_est_nom.accm_bias + rov_est_err.mean.accm_bias
        gyro_bias_inj = rov_est_nom.gyro_bias + rov_est_err.mean.gyro_bias

        x_nom_inj = NominalState(
            pos_inj, vel_inj, ori_inj, accm_bias_inj, gyro_bias_inj
        )

        G = np.eye(15)
        G[6:9, 6:9] -= get_cross_matrix(0.5 * rov_est_err.mean.avec)
        P_inj = G @ rov_est_err.cov @ G.T
        x_err_inj = MultiVarGauss[ErrorState](np.zeros(15), P_inj)
        return EskfState(x_nom_inj, x_err_inj)


@dataclass
class ESKF_imu(_ESKFUpdateShared):
    model: ModelIMU
    sensorUsbl: Optional[SensorUSBL] = None
    sensorRange: Optional[SensorRange] = None
    sensorDepth: Optional[SensorDepth] = None

    def predict_from_imu(self,
                         x_est_prev: EskfState,
                         z_imu: ImuMeasurement,
                         dt: float
                         ) -> EskfState:
        """Method called every time an IMU measurement is received

        Args:
            x_nom_prev: previous eskf state
            z_imu: raw IMU measurement
            dt: time since last imu measurement
        Returns:
            x_est_pred: predicted eskf state
        """
        if dt == 0:
            return x_est_prev

        x_est_prev_nom = x_est_prev.nom
        z_corr = self.model.correct_z_imu(x_est_prev_nom, z_imu)
        x_est_pred_nom = self.model.predict_nom(x_est_prev_nom, z_corr, dt) 
        x_est_pred_err = self.model.predict_err(x_est_prev, z_corr, dt)

        x_est_pred = EskfState(x_est_pred_nom, x_est_pred_err)

        return x_est_pred

    def update_from_usbl(self, 
                        rov_est_pred: EskfState, 
                        asv_state: ASVState, 
                        z_usbl: UsblMeasurement) -> Tuple[EskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorUsbl, "sensorUsbl")
        z_est_pred = sensor.pred_from_est(rov_est_pred, asv_state)
        H = sensor.H(rov_est_pred.nom, asv_state)
        R = sensor.R

        upd_err = self._update_err(
            rov_est_pred, z_est_pred, z_usbl, H, R, is_usbl=True
        )
        return self._inject(rov_est_pred.nom, upd_err), z_est_pred
    
    def update_from_range(self, 
                            rov_est_pred: EskfState, 
                            asv_state: ASVState, 
                            z_range: RangeMeasurement) -> Tuple[EskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorRange, "sensorRange")
        z_est_pred = sensor.pred_from_est(rov_est_pred, asv_state)
        H = sensor.H(rov_est_pred.nom, asv_state)
        R = sensor.R

        upd_err = self._update_err(rov_est_pred, z_est_pred, z_range, H, R)
        return self._inject(rov_est_pred.nom, upd_err), z_est_pred

    def update_from_depth(self, 
                          rov_est_pred: EskfState, 
                          z_depth: DepthMeasurement) -> Tuple[EskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorDepth, "sensorDepth")
        z_est_pred = sensor.pred_from_est(rov_est_pred)
        H = sensor.H(rov_est_pred.nom)
        R = sensor.R

        upd_err = self._update_err(rov_est_pred, z_est_pred, z_depth, H, R)
        return self._inject(rov_est_pred.nom, upd_err), z_est_pred



@ dataclass
class ESKF_cv(_ESKFUpdateShared):
    modelCv: ModelCV
    sensorUsbl: Optional[SensorUSBL] = None
    sensorRange: Optional[SensorRange] = None
    sensorDepth: Optional[SensorDepth] = None

    def predict_with_cv(self,
                   x_est_prev: EskfState,
                   dt: float
                   ) -> EskfState:
        """Predict the nominal and error state using a constant velocity model.
        Used for testing and as a baseline.

        Args:
            x_est_prev: previous eskf state
            dt: time step
        Returns:             x_est_pred: predicted eskf state
        """
        if dt == 0:
            return x_est_prev

        x_est_prev_nom = x_est_prev.nom
        x_est_pred_nom = self.modelCv.predict_nom(x_est_prev_nom, dt) 
        x_est_pred_err = self.modelCv.predict_err(x_est_prev, dt)

        x_est_pred = EskfState(x_est_pred_nom, x_est_pred_err)

        return x_est_pred

    def update_from_usbl(self, 
                        rov_est_pred: EskfState, 
                        asv_state: ASVState, 
                        z_usbl: UsblMeasurement) -> Tuple[EskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorUsbl, "sensorUsbl")
        z_est_pred = sensor.pred_from_est(rov_est_pred, asv_state)
        H = sensor.H(rov_est_pred.nom, asv_state)
        R = sensor.R

        upd_err = self._update_err(
            rov_est_pred, z_est_pred, z_usbl, H, R, is_usbl=True
        )
        return self._inject(rov_est_pred.nom, upd_err), z_est_pred
    
    def update_from_range(self, 
                            rov_est_pred: EskfState, 
                            asv_state: ASVState, 
                            z_range: RangeMeasurement) -> Tuple[EskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorRange, "sensorRange")
        z_est_pred = sensor.pred_from_est(rov_est_pred, asv_state)
        H = sensor.H(rov_est_pred.nom, asv_state)
        R = sensor.R

        upd_err = self._update_err(rov_est_pred, z_est_pred, z_range, H, R)
        return self._inject(rov_est_pred.nom, upd_err), z_est_pred

    def update_from_depth(self, 
                          rov_est_pred: EskfState, 
                          z_depth: DepthMeasurement) -> Tuple[EskfState, MultiVarGauss]:
        sensor = self._require_sensor(self.sensorDepth, "sensorDepth")
        z_est_pred = sensor.pred_from_est(rov_est_pred)
        H = sensor.H(rov_est_pred.nom)
        R = sensor.R

        upd_err = self._update_err(rov_est_pred, z_est_pred, z_depth, H, R)
        return self._inject(rov_est_pred.nom, upd_err), z_est_pred
