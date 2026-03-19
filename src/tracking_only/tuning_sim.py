from senfuslib import MultiVarGauss
import numpy as np

from quaternion import RotationQuaterion

from tracking_only.eskf import ESKF_cv
from tracking_only.models import ModelCV
from tracking_only.states import RovNominalState, RovErrorState, RovEskfState
from tracking_only.sensors import SensorUSBL, SensorRange, SensorDepth



usbl_lever_arm = np.array([0, 0, 1.2])  # Lever arm of the USBL sensor in meters (x, y, z) relative to the ASV's center of mass


start_time_sim = 0.  # Start time, set to None for full time
end_time_sim = 300  # End time in seconds, set to None to use all data


cv_sim = ModelCV(
    sigma_a=0.01,  # Acceleration standard deviation for the CV model
)


usbl_sim = SensorUSBL(
    usbl_std=np.deg2rad(1),  # USBL measurement standard deviation
    lever_arm=usbl_lever_arm,  # sensor position relative to origin
)

range_sim = SensorRange(
    range_std=0.5,  # Range measurement standard deviation in meters
    lever_arm=usbl_lever_arm,  # sensor position relative to origin
)

depth_sim = SensorDepth(
    depth_std = 0.5,
)

# Accm and gyro bias zero for CV-model
rov_est_init_nom_sim = RovNominalState(
    pos=np.array([0.0, 0.0, 5.0]),  # matches ROV trajectory start
    vel=np.array([0.5, 0.0, 0.0]),  # realistic ROV speed ~0.5 m/s
    ori=RotationQuaterion.from_euler([0, 0, 0]),
    accm_bias=np.zeros(3),
    gyro_bias=np.zeros(3),
)

rov_err_init_std_sim = np.repeat(repeats=3, a=[
    2,  # position
    0.1,  # velocity
    np.deg2rad(5),  # angle vector
    0.0,  # accelerometer bias
    0.00  # gyro bias
])


rov_est_init_err_sim = MultiVarGauss[RovErrorState]( 
    np.zeros(15), 
    np.diag(rov_err_init_std_sim**2)) 


eskf_sim = ESKF_cv(cv_sim, usbl_sim, range_sim, depth_sim) 
rov_est_init_sim = RovEskfState(rov_est_init_nom_sim, rov_est_init_err_sim)