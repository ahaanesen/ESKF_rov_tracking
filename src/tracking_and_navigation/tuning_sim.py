import numpy as np
from senfuslib import MultiVarGauss

from quaternion import RotationQuaterion

from tracking_and_navigation.eskf import ESKF_joint
from tracking_and_navigation.models import ModelCV, ModelIMU
from tracking_and_navigation.sensors import (
    SensorGNSS_ASV,
    SensorUSBL_Joint,
    SensorRange_Joint,
    SensorDepth_ROV,
)
from tracking_and_navigation.states import (
    JointEskfState,
    JointIdx,
    JointNominalState,
    JointErrorState,
    ASVNominalState,
    ROVNominalCV,
)

# -----------------------------------------------------------------------------
# Simulation time
# -----------------------------------------------------------------------------
start_time_sim = 0.0
end_time_sim = 300.0

# -----------------------------------------------------------------------------
# Lever arms (ASV body frame -> NED through ASV orientation)
# -----------------------------------------------------------------------------
gnss_lever_arm = np.array([0.3, 0.3, 0.1])
usbl_lever_arm = np.array([0.0, 0.0, 1.2])

# -----------------------------------------------------------------------------
# IMU correction matrices (example)
# -----------------------------------------------------------------------------
accm_corr = np.diag([1.01, 0.98, 1.02])
gyro_corr = np.diag([0.99, 1.02, 1.00])

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
imu_sim = ModelIMU(
    accm_std=1.167e-3,
    accm_bias_std=4e-3,
    accm_bias_p=1e-16,
    gyro_std=4.36e-5,
    gyro_bias_std=5e-5,
    gyro_bias_p=1e-16,
    accm_correction=accm_corr,
    gyro_correction=gyro_corr,
)

cv_sim = ModelCV(
    sigma_a=0.20,
)

# -----------------------------------------------------------------------------
# Sensors
# -----------------------------------------------------------------------------
gnss_sim = SensorGNSS_ASV(
    gnss_std_ne=0.3,
    gnss_std_d=0.5,
    lever_arm=gnss_lever_arm,
)

# IMPORTANT: match your sensor constructor args.
# If your class uses lever_arm_asv, keep that. If it uses lever_arm, rename.
usbl_sim = SensorUSBL_Joint(
    usbl_std=np.deg2rad(1),
    lever_arm=usbl_lever_arm,
)

range_sim = SensorRange_Joint(
    range_std=0.5,
    lever_arm=usbl_lever_arm,
)

depth_sim = SensorDepth_ROV(
    depth_std=0.3,
)

# -----------------------------------------------------------------------------
# Initial estimate (Joint state)
# -----------------------------------------------------------------------------
asv_est_init_nom_sim = ASVNominalState(
    pos=np.array([50.0, 0.0, 0.0]),
    vel=np.array([0.0, 0.0, 0.0]),
    ori=RotationQuaterion.from_euler([0.0, 0.0, 0.0]),
    accm_bias=np.zeros(3),
    gyro_bias=np.zeros(3),
)

# ASV error stds: [pos(3), vel(3), avec(3), acc_bias(3), gyro_bias(3)] => 15
asv_err_init_std_sim = np.repeat(
    repeats=3,
    a=[
        1.0,               # pos
        0.1,               # vel
        np.deg2rad(5.0),   # attitude error vector
        0.01,              # acc bias
        0.001,             # gyro bias
    ],
)

rov_est_init_nom_sim = ROVNominalCV(
    pos=np.array([0.0, 0.0, 5.0]),
    vel=np.array([0.5, 0.0, 0.0]),
)

# ROV CV error stds: [pos(3), vel(3)] => 6
rov_err_init_std_sim = np.repeat(
    repeats=3,
    a=[
        2.0,    # pos
        0.1,    # vel
    ],
)

P0 = np.diag(np.concatenate((asv_err_init_std_sim, rov_err_init_std_sim)) ** 2)

x_init_err_sim = MultiVarGauss[JointErrorState](
    JointErrorState.from_array(np.zeros(JointIdx.N)),
    P0,
)

x_init_sim = JointEskfState(
    nom=JointNominalState(asv=asv_est_init_nom_sim, rov=rov_est_init_nom_sim),
    err=x_init_err_sim,
)

# -----------------------------------------------------------------------------
# Filter instance
# -----------------------------------------------------------------------------
eskf_sim = ESKF_joint(
    modelImuAsv=imu_sim,
    modelCvRov=cv_sim,
    sensorGnssAsv=gnss_sim,
    sensorUsbl=usbl_sim,
    sensorRange=range_sim,
    sensorDepth=depth_sim,
)