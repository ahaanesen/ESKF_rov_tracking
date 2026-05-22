import numpy as np

from quaternion import RotationQuaterion
from tracking_and_navigation.generate_trajectories import generate_trajectories, TrajectoryType
from tracking_and_navigation.generate_measurements import MeasurementGenerator

from tracking_and_navigation.run_eskf import run_eskf_s1, run_eskf_s2, run_eskf_s3
from tracking_and_navigation.plotting import PlotterESKFJoint
from tracking_and_navigation.states import AsvNominalState, JointEskfState, JointNominalState

from tracking_and_navigation.tuning_sim import (
    eskf_sim,
    x_init_sim,
    usbl_sim,
    range_sim,
    depth_sim,
    gnss_sim,
    imu_sim,
)

# =============================================================================
# Scenario configuration — edit here to switch between modes
# =============================================================================
#
# TRAJECTORY_TYPE controls the USV/ROV motion geometry:
#   TrajectoryType.CIRCULAR   — circular ASV, piecewise-linear ROV (ideal for ESKF)
#   TrajectoryType.FIGURE_8   — lemniscate ASV, curved ROV (better bearing geometry)
#   TrajectoryType.SINUSOIDAL — S-curve ASV, maneuvering ROV (hardest for ESKF)
#
TRAJECTORY_TYPE = TrajectoryType.CIRCULAR

#
# Measurement realism settings (all False/0 = ideal, synchronous measurements):
#   ACOUSTIC_DELAY — shift reception timestamp by one-way TOF = range/SOUND_SPEED
#                    (the key scenario where FGO's delayed-measurement handling helps)
#   JITTER_STD     — Gaussian timing jitter std [s] on acoustic reception
#   MISS_PROB      — probability of dropping each acoustic measurement [0, 1]
#   SOUND_SPEED    — acoustic propagation speed [m/s]
#
ACOUSTIC_DELAY = True
JITTER_STD     = 0.1    # ±500 ms 1-sigma
MISS_PROB      = 0.10    # 10 % dropout
SOUND_SPEED    = 1500.0  # m/s
# JITTER_STD     = 0.0
# MISS_PROB      = 0.0
# SOUND_SPEED    = 1500.0


def run_simulations():

    # 1) Ground truth
    asv_gt_tseq, rov_gt_tseq, _ = generate_trajectories(
        duration=300,
        dt=0.01,
        trajectory_type=TRAJECTORY_TYPE,
    )

    # 2) Measurements
    gen = MeasurementGenerator(asv_gt_tseq, rov_gt_tseq)

    z_imu_tseq = gen.generate_imu_asv(
        accm_std=imu_sim.accm_std,
        gyro_std=imu_sim.gyro_std,
        rate_hz=100.0,
    )
    z_gnss_tseq = gen.generate_gnss_asv(
        std_ne=gnss_sim.gnss_std_ne,
        std_d=gnss_sim.gnss_std_d,
        lever_arm=gnss_sim.lever_arm,
        rate_hz=1.0,
    )
    z_usbl_tseq = gen.generate_usbl(
        std_rad=usbl_sim.usbl_std,
        lever_arm=usbl_sim.lever_arm,
        rate_hz=0.2,
        acoustic_delay=ACOUSTIC_DELAY,
        jitter_std=JITTER_STD,
        miss_prob=MISS_PROB,
        sound_speed=SOUND_SPEED,
    )
    z_range_tseq = gen.generate_range(
        std_m=range_sim.range_std,
        lever_arm=range_sim.lever_arm,
        rate_hz=0.2,
        acoustic_delay=ACOUSTIC_DELAY,
        jitter_std=JITTER_STD,
        miss_prob=MISS_PROB,
        sound_speed=SOUND_SPEED,
    )
    z_depth_tseq = gen.generate_depth(
        std_m=depth_sim.depth_std,
        rate_hz=0.2,
        miss_prob=0.0,
    )

    x_init = _init_asv_from_gnss(x_init_sim, z_gnss_tseq)

    traj_name = TRAJECTORY_TYPE.value
    save_dir = f"plots/{traj_name}_dt0_01_cv_020_aco_delay_{ACOUSTIC_DELAY}_jitter_{JITTER_STD}_miss_{MISS_PROB}"

    # 3) Run scenarios
    print(f"[{traj_name}] Scenario 1: GNSS + Bearing only")
    upd_s1, pred_s1 = run_eskf_s1(
        eskf=eskf_sim,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        z_usbl_tseq=z_usbl_tseq,
    )

    print(f"[{traj_name}] Scenario 2: GNSS + Bearing + Range")
    upd_s2, pred_s2 = run_eskf_s2(
        eskf=eskf_sim,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        z_usbl_tseq=z_usbl_tseq,
        z_range_tseq=z_range_tseq,
    )

    print(f"[{traj_name}] Scenario 3: GNSS + Bearing + Range + Depth")
    upd_s3, pred_s3 = run_eskf_s3(
        eskf=eskf_sim,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        z_usbl_tseq=z_usbl_tseq,
        z_range_tseq=z_range_tseq,
        z_depth_tseq=z_depth_tseq,
    )

    # 4) Plot
    PlotterESKFJoint(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_gt_tseq,
        x_upds=upd_s1,
        x_preds=pred_s1,
        z_gnss_asv=z_gnss_tseq,
        z_usbl=z_usbl_tseq,
        scenario_name=f"[{traj_name}] Scenario 1: Bearing-only",
        save_dir=f"{save_dir}/scenario1",
    ).show()

    PlotterESKFJoint(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_gt_tseq,
        x_upds=upd_s2,
        x_preds=pred_s2,
        z_gnss_asv=z_gnss_tseq,
        z_usbl=z_usbl_tseq,
        z_range=z_range_tseq,
        scenario_name=f"[{traj_name}] Scenario 2: Bearing + Range",
        save_dir=f"{save_dir}/scenario2",
    ).show()

    PlotterESKFJoint(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_gt_tseq,
        x_upds=upd_s3,
        x_preds=pred_s3,
        z_gnss_asv=z_gnss_tseq,
        z_usbl=z_usbl_tseq,
        z_range=z_range_tseq,
        z_depth=z_depth_tseq,
        scenario_name=f"[{traj_name}] Scenario 3: Bearing + Range + Depth",
        save_dir=f"{save_dir}/scenario3",
    ).show()


def _init_asv_from_gnss(
    x_init: JointEskfState,
    z_gnss_tseq,
) -> JointEskfState:
    gnss_items = list(z_gnss_tseq.items())
    if len(gnss_items) < 2:
        return x_init

    _, z0 = gnss_items[0]
    _, z1 = gnss_items[1]

    pos0 = np.asarray(z0.pos, dtype=float).reshape(3)
    pos1 = np.asarray(z1.pos, dtype=float).reshape(3)
    d_ne = pos1[:2] - pos0[:2]

    if np.hypot(d_ne[0], d_ne[1]) < 1e-9:
        yaw = x_init.nom.asv.euler[2]
    else:
        yaw = float(np.arctan2(d_ne[1], d_ne[0]))

    asv_prev = x_init.nom.asv
    asv_init = AsvNominalState(
        pos=pos1,
        vel=[0,0,0],
        ori=RotationQuaterion.from_euler([0.0, 0.0, yaw]),
        accm_bias=asv_prev.accm_bias,
        gyro_bias=asv_prev.gyro_bias,
    )
    return JointEskfState(
        nom=JointNominalState(asv=asv_init, rov=x_init.nom.rov),
        err=x_init.err,
    )
