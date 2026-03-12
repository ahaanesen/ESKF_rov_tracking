from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from senfuslib import MultiVarGauss
from eskf.src.rov_states import NominalState, EskfState
from eskf.src.asv_states import ASVState, UsblMeasurement, RangeMeasurement
from utils.cross_matrix import get_cross_matrix


@ dataclass
class SensorUSBL:
    usbl_std: float
    lever_arm: 'np.ndarray[3]'
    R: 'np.ndarray[2, 2]' = field(init=False)

    def __post_init__(self):
        self.R = np.array([[self.usbl_std**2], 
                           [self.usbl_std**2]])

    def H(self, rov_nom: NominalState, asv_state: ASVState) -> 'np.ndarray[2, 15]':
        """Get the measurement jacobian, H with respect to the error state.

        Hint: the usbl sensor has a relative position to the center of the ASV given by
        self.lever_arm. How will the usbl measurement change if the drone is 
        rotated differently? Use get_cross_matrix and some other stuff. 
¨
        Returns:
            H (ndarray[2, 15]): the measurement matrix
        """
        H = np.zeros((2, 15))

        # 1. Calculate relative vector in Navigation Frame
        R_asv = asv_state.ori.as_rotmat()
        pos_usbl = asv_state.pos + R_asv @ self.lever_arm

        d = rov_nom.pos - pos_usbl
        dx, dy, dz = d

        # Horizontal range and total slant range
        r_sq = dx**2 + dy**2
        r = np.sqrt(r_sq)
        rho_sq = r_sq + dz**2
        rho = np.sqrt(rho_sq)

        # Singularities: r=0 (zenith/nadir) or rho=0 (same point)
        if r < 1e-6 or rho_sq < 1e-9:
            return H

        # 2. dh/dd: Jacobian of measurements w.r.t the relative vector d
        # Azimuth: atan2(dy, dx)
        # Elevation: atan2(dz, r)
        dh_dd = np.array([
            [-dy/r_sq,             dx/r_sq,             0],       # Azimuth row
            [-(dx*dz)/(rho_sq*r),  -(dy*dz)/(rho_sq*r),  r/rho_sq] # Elevation row
        ])

        # 3. Fill H matrix
        # Position error state (indices 0:3)
        H[:3, :3] = dh_dd @ np.eye(3)

        # Orientation error state (indices 6:9)
        # Cannot estimate ROV orientation from USBL 

        return H
    
    def pred_from_est(self, rov_est: EskfState, asv_state: ASVState
                      ) -> MultiVarGauss[UsblMeasurement]:
        """Predict the usbl measurement

        Args:
            x_est: eskf state

        Returns:
            z_usbl_pred_gauss: usbl prediction gaussian
        """
        rov_est_nom = rov_est.nom
        rov_est_err = rov_est.err

        P = rov_est_err.cov

        usbl_pos = asv_state.pos + asv_state.ori.as_rotmat() @ self.lever_arm
        d = rov_est_nom.pos - usbl_pos
        dx, dy, dz = d

        azimuth = np.arctan2(dy, dx)
        elevation = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        z_pred = np.array([azimuth, elevation])
        S = self.R + self.H(rov_est_nom, asv_state) @ P @ self.H(rov_est_nom, asv_state).T


        z_pred = UsblMeasurement.from_array(np.array([z_pred]))
        z_usbl_pred_gauss = MultiVarGauss[UsblMeasurement](z_pred, S)

        return z_usbl_pred_gauss
    
@ dataclass
class SensorRange:
    range_std: float
    lever_arm: 'np.ndarray[3]'
    R: 'np.ndarray[1, 1]' = field(init=False)

    def __post_init__(self):
        self.R = np.array([[self.range_std**2]])
    
    def H(self, rov_nom: NominalState, asv_state: ASVState) -> 'np.ndarray[1, 15]':
        """Get the measurement jacobian, H with respect to the error state.

         Returns:
            H (ndarray[1, 15]): the measurement matrix
        """
        H = np.zeros((1, 15))

        # 1. Calculate relative vector in Navigation Frame
        R_asv = asv_state.ori.as_rotmat()
        pos_sensor = asv_state.pos + R_asv @ self.lever_arm
        d = rov_nom.pos - pos_sensor
        dx, dy, dz = d

        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Handle the singularity at zero range
        if r < 1e-6:
            return H 

        # 2. dr/dd: Jacobian of range w.r.t the relative vector d
        dr_dd = np.array([[dx/r, dy/r, dz/r]])

        # 3. Fill H matrix
        # Position error state (indices 0:3)
        H[:1, :3] = dr_dd @ np.eye(3)

        # Orientation error state (indices 6:9)
        # Cannot estimate ROV orientation from range 

        return H
    
    def pred_from_est(self, rov_est: EskfState, asv_state: ASVState
                      ) -> MultiVarGauss[RangeMeasurement]:
        """Predict the range measurement

        Args:
            rov_est: eskf state
            asv_state: asv state

        Returns:
            z_range_pred_gauss: range prediction gaussian
        """
        rov_est_nom = rov_est.nom
        rov_est_err = rov_est.err

        P = rov_est_err.cov

        range_pos = asv_state.pos + asv_state.ori.as_rotmat() @ self.lever_arm
        d = rov_est_nom.pos - range_pos
        dx, dy, dz = d

        range_pred = np.sqrt(dx**2 + dy**2 + dz**2)
        z_pred = np.array([range_pred])
        S = self.R + self.H(rov_est_nom, asv_state) @ P @ self.H(rov_est_nom, asv_state).T


        z_pred = RangeMeasurement.from_array(np.array([z_pred]))
        z_range_pred_gauss = MultiVarGauss[RangeMeasurement](z_pred, S)

        return z_range_pred_gauss