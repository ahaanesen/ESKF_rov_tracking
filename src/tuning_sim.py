from eskf import ESKF_cv
from models import ModelIMU, ModelCV
from rov_sensors import SensorDepth
from asv_sensors import SensorRange, SensorUSBL
from rov_states import EskfState, NominalState, ErrorState, RotationQuaterion
from asv_states import ASVState

from utils.dataloader import load_drone_params
from senfuslib import MultiVarGauss
import numpy as np
from config import fname_data_sim, fname_data_real

accm_corr, gyro_corr, imu_lever_arm = load_drone_params(fname_data_sim)
usbl_lever_arm = np.array([0, 0, 1.2])  # Lever arm of the USBL sensor in meters (x, y, z) relative to the ASV's center of mass

"""Everything below here can be altered"""
start_time_sim = 0.  # Start time, set to None for full time
end_time_sim = 300  # End time in seconds, set to None to use all data

# imu_min_dt_sim = None  # IMU is sampled at 100 Hz, use to downsample
# gnss_min_dt_sim = None  # GPS is sampled at 1 Hz, use this to downsample

imu_sim = ModelIMU(
    accm_std=1.167e-3,   # Accelerometer standard deviation, TUNABE
    accm_bias_std=4e-3,  # Accelerometer bias standard deviation
    accm_bias_p=1e-16,  # Accelerometer inv time constant see (10.57)

    gyro_std=4.36e-5,  # Gyro standard deviation
    gyro_bias_std=5e-5,  # Gyro bias standard deviation
    gyro_bias_p=1e-16,  # Gyro inv time constant see (10.57)

    accm_correction=accm_corr,  # Accelerometer correction matrix
    gyro_correction=gyro_corr,  # Gyro correction matrix
)

cv_sim = ModelCV(
    sigma_a=0.01,  # Acceleration standard deviation for the CV model
)

# gnss_sim = SensorGNSS(
#     gnss_std_ne=0.3,  # GNSS standard deviation in North and East
#     gnss_std_d=0.5,  # GNSS standard deviation in Down
#     lever_arm=lever_arm,  # antenna position relative to origin
# )

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


# x_est_init_nom_sim = NominalState(
#     pos=np.array([0.2, 0, -5]),  # position
#     vel=np.array([20, 0, 0]),  # velocity
#     ori=RotationQuaterion.from_euler([0, 0, 0]),  # orientation
#     accm_bias=np.zeros(3),  # accelerometer bias
#     gyro_bias=np.zeros(3),  # gyro bias
# )

# x_err_init_std_sim = np.repeat(repeats=3, a=[
#     2,  # position
#     0.1,  # velocity
#     np.deg2rad(5),  # angle vector
#     0.01,  # accelerometer bias
#     0.001  # gyro bias
# ])

# Accm and gyro bias zero for CV-model
rov_est_init_nom_sim = NominalState(
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

# """Dont change anything below here"""
# x_est_init_err_sim = MultiVarGauss[ErrorState](  # Don't change this
#     np.zeros(15),  # Don't change this
#     np.diag(x_err_init_std_sim**2))  # Don't change this


# eskf_sim = ESKF_imu(imu_sim, gnss_sim)  # Don't change this
# x_est_init_sim = EskfState(x_est_init_nom_sim, x_est_init_err_sim)

rov_est_init_err_sim = MultiVarGauss[ErrorState](  # Don't change this
    np.zeros(15),  # Don't change this
    np.diag(rov_err_init_std_sim**2))  # Don't change this


eskf_sim = ESKF_cv(cv_sim, usbl_sim, range_sim, depth_sim)  # Don't change this
rov_est_init_sim = EskfState(rov_est_init_nom_sim, rov_est_init_err_sim)