import numpy as np

import scipy.linalg
from dataclasses import dataclass, field
from typing import Tuple

from senfuslib import MultiVarGauss
from eskf.src.rov_states import (ImuMeasurement,

                    GnssMeasurement, EskfState)
from eskf.src.rov_states import NominalState, ErrorState
from eskf.src.rov_sensors import SensorGNSS, SensorDepth
from eskf.src.asv_sensors import SensorUSBL, SensorRange
from eskf.src.asv_states import ASVState, UsblMeasurement

from quaternion import RotationQuaterion
from utils.cross_matrix import get_cross_matrix
from models import ModelIMU, ModelCV


@dataclass
class ESKF_imu():
    model: ModelIMU
    sensor: SensorGNSS

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

    def update_err_from_gnss(self,
                             x_est_pred: EskfState,
                             z_est_pred: MultiVarGauss[GnssMeasurement],
                             z_gnss: GnssMeasurement
                             ) -> MultiVarGauss[ErrorState]:
        """Update the error state from a gnss measurement

        Hint: see (10.75)
        Due to numerical error its recomended use the robust calculation of 
        posterior covariance, somtimes called Joseph form:
            I_WH = np.eye(*P.shape) - W @ H
            x_err_cov_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)
        Remember that:
            S = H @ P @ H.T + R
        and that:
            np.linalg.solve(S, H.T) is faster than np.linalg.inv(S)

        Args:
            x_est_pred: predicted nominal and error state (gaussian)
            z_est_pred: predicted gnss measurement (gaussian)
            z_gnss: gnss measurement

        Returns:
            x_est_upd_err: updated error state gaussian
        """
        x_nom = x_est_pred.nom
        x_err = x_est_pred.err
        z_pred, S = z_est_pred

        innovation = z_gnss - z_pred 
        H = self.sensor.H(x_nom)
        P = x_err.cov
        R = self.sensor.R
        W = np.linalg.solve(S.T, H @ P.T).T 
        x_err_upd = W @ innovation
        I_WH = np.eye(*P.shape) - W @ H 
        x_err_cov_upd = I_WH @ P @ I_WH.T + W @ R @ W.T

        x_err_upd = ErrorState.from_array(x_err_upd)
        x_est_upd_err = MultiVarGauss[ErrorState](x_err_upd, x_err_cov_upd)

        return x_est_upd_err

    def inject(self,
               x_est_nom: NominalState,
               x_est_err: MultiVarGauss[ErrorState],
               ) -> EskfState:
        """Perform the injection step

        Hint: see (10.85) and (10.72) on how to inject into nominal state.
        See (10.86) on how to find error covariance after injection

        (10.85): x <- x (+) delta x_hat
        (10.86): P <- G @ P @ G.T

        Args:
            x_nom_prev: previous nominal state
            x_err_upd: updated error state gaussian

        Returns:
            x_est_inj: eskf state after injection
        """
        # (10.72)
        pos_inj = x_est_nom.pos + x_est_err.mean.pos
        vel_inj = x_est_nom.vel + x_est_err.mean.vel
        ori_inj = RotationQuaterion(1, np.zeros(3))
        ori_inj = x_est_nom.ori.multiply(RotationQuaterion(1, 0.5*x_est_err.mean.avec))
        accm_bias_inj = x_est_nom.accm_bias + x_est_err.mean.accm_bias
        gyro_bias_inj = x_est_nom.gyro_bias + x_est_err.mean.gyro_bias

        x_nom_inj = NominalState(pos_inj, vel_inj, ori_inj,
                                 accm_bias_inj, gyro_bias_inj)

        G = np.eye(15)
        G[6:9, 6:9] -= get_cross_matrix(0.5 * x_est_err.mean.avec)
        P_inj = G @ x_est_err.cov @ G.T
        x_err_inj = MultiVarGauss[ErrorState](np.zeros(15), P_inj)
        x_est_inj = EskfState(x_nom_inj, x_err_inj)

        return x_est_inj

    def update_from_gnss(self,
                         x_est_pred: EskfState,
                         z_gnss: GnssMeasurement,
                         ) -> Tuple[NominalState,
                                    MultiVarGauss[ErrorState],
                                    MultiVarGauss]:
        """Method called every time an gnss measurement is received.


        Args:
            x_est_pred: previous estimated esfk state
            z_gnss: gnss measurement

        Returns:
            x_est_upd: updated eskf state
            z_est_upd: predicted measurement gaussian

        """
        z_est_pred = self.sensor.pred_from_est(x_est_pred)  # TODO
        x_est_upd_err = self.update_err_from_gnss(x_est_pred, z_est_pred, z_gnss)  # TODO
        x_est_upd = self.inject(x_est_pred.nom, x_est_upd_err)  # TODO

        return x_est_upd, z_est_pred

@ dataclass
class ESKF_cv():
    modelCv: ModelCV
    sensorUsbl: SensorUSBL
    sensorRange: SensorRange
    sensorDepth: SensorDepth

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
    
    def _update_err(self, 
                rov_est_pred: EskfState,
                z_pred_gauss: MultiVarGauss,
                z_measurement: np.ndarray,
                H: np.ndarray,
                R: np.ndarray,
                is_usbl: bool = False # Flag to trigger wrapping
                ) -> MultiVarGauss[ErrorState]:
        
        P = rov_est_pred.err.cov
        z_pred, S = z_pred_gauss
        
        # 1. Calculate Innovation
        innovation = z_measurement - z_pred 
        
        # 2. Angle Wrapping for USBL (Azimuth is at index 0)
        if is_usbl:
            # Wrap Azimuth to [-pi, pi]
            innovation[0] = (innovation[0] + np.pi) % (2 * np.pi) - np.pi
            # Note: Elevation (index 1) typically stays within [-pi/2, pi/2]
            # and does not require wrapping in standard setups.

        # 3. Kalman Gain Calculation
        W = np.linalg.solve(S.T, H @ P.T).T 
        
        # 4. Update Error State Mean
        rov_err_upd_mean = W @ innovation
        
        # 5. Update Error State Covariance (Joseph Form)
        I_WH = np.eye(15) - W @ H 
        rov_err_cov_upd = I_WH @ P @ I_WH.T + W @ R @ W.T

        return MultiVarGauss[ErrorState](ErrorState.from_array(rov_err_upd_mean), rov_err_cov_upd)

    def _inject(self,
               rov_est_nom: NominalState,
               rov_est_err: MultiVarGauss[ErrorState],
               ) -> EskfState:
        """Perform the injection step

        Hint: see (10.85) and (10.72) on how to inject into nominal state.
        See (10.86) on how to find error covariance after injection

        (10.85): x <- x (+) delta x_hat
        (10.86): P <- G @ P @ G.T

        Args:
            rov_est_nom: previous nominal state
            rov_est_err: updated error state gaussian

        Returns:
            x_est_inj: eskf state after injection
        """
        # (10.72)
        pos_inj = rov_est_nom.pos + rov_est_err.mean.pos
        vel_inj = rov_est_nom.vel + rov_est_err.mean.vel
        ori_inj = RotationQuaterion(1, np.zeros(3))
        ori_inj = rov_est_nom.ori.multiply(RotationQuaterion(1, 0.5*rov_est_err.mean.avec))
        accm_bias_inj = rov_est_nom.accm_bias + rov_est_err.mean.accm_bias
        gyro_bias_inj = rov_est_nom.gyro_bias + rov_est_err.mean.gyro_bias

        x_nom_inj = NominalState(pos_inj, vel_inj, ori_inj,
                                 accm_bias_inj, gyro_bias_inj)

        G = np.eye(15)
        G[6:9, 6:9] -= get_cross_matrix(0.5 * rov_est_err.mean.avec)
        P_inj = G @ rov_est_err.cov @ G.T
        x_err_inj = MultiVarGauss[ErrorState](np.zeros(15), P_inj)
        x_est_inj = EskfState(x_nom_inj, x_err_inj)

        return x_est_inj

    def update_from_usbl(self, 
                        rov_est_pred: EskfState, 
                        asv_state: ASVState, 
                        z_usbl: UsblMeasurement) -> Tuple[EskfState, MultiVarGauss]:
        
        z_est_pred = self.sensorUsbl.pred_from_est(rov_est_pred, asv_state)
        H = self.sensorUsbl.H(rov_est_pred.nom, asv_state)
        R = self.sensorUsbl.R
        
        # Convert z_usbl to array if it isn't already for the innovation math
        z_meas_arr = np.array(z_usbl) 
        
        upd_err = self._update_err(rov_est_pred, z_est_pred, z_meas_arr, H, R, is_usbl=True)
        
        return self._inject(rov_est_pred.nom, upd_err), z_est_pred
    
    def update_from_range(self, 
                            rov_est_pred: EskfState, 
                            asv_state: ASVState, 
                            z_range: float) -> Tuple[EskfState, MultiVarGauss]:
            """Complete Range update cycle."""
            # Note: Assuming your sensorRange has similar pred_from_est and H methods
            z_est_pred = self.sensorRange.pred_from_est(rov_est_pred, asv_state)
            
            H = self.sensorRange.H(rov_est_pred.nom, asv_state)
            R = self.sensorRange.R
            
            upd_err = self._update_err(rov_est_pred, z_est_pred, z_range, H, R)
            return self._inject(rov_est_pred.nom, upd_err), z_est_pred

    def update_from_depth(self, 
                          rov_est_pred: EskfState, 
                          z_depth: float) -> Tuple[EskfState, MultiVarGauss]:
        """Complete Depth update cycle."""
        # Depth is simple; usually measured directly as Z in NED
        z_est_pred = self.sensorDepth.pred_from_est(rov_est_pred)
        
        H = self.sensorDepth.H(rov_est_pred.nom)
        R = self.sensorDepth.R
        
        upd_err = self._update_err(rov_est_pred, z_est_pred, z_depth, H, R)
        return self._inject(rov_est_pred.nom, upd_err), z_est_pred
