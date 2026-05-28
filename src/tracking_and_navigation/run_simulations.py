import csv
from pathlib import Path

import numpy as np

from quaternion import RotationQuaterion
from senfuslib.gaussian import MultiVarGauss
from senfuslib.gaussian import MultiVarGauss
from tracking_and_navigation.generate_trajectories import generate_trajectories, TrajectoryType
from tracking_and_navigation.generate_measurements import MeasurementGenerator

from tracking_and_navigation.run_eskf import run_eskf_s1, run_eskf_s2, run_eskf_s3
from tracking_and_navigation.plotting import PlotterESKFJoint
from tracking_and_navigation.states import AsvNominalState, JointErrorState, JointEskfState, JointIdx, JointNominalState, RovNominalCV
from utils.withXYZ import WithXYZ

from tracking_and_navigation.tuning_sim import (
    eskf_sim,
    x_init_sim,

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
# TRAJECTORY_TYPE = TrajectoryType.FIGURE_8

# #
# # Measurement realism settings (all False/0 = ideal, synchronous measurements):
# #   ACOUSTIC_DELAY — shift reception timestamp by one-way TOF = range/SOUND_SPEED
# #                    (the key scenario where FGO's delayed-measurement handling helps)
# #   JITTER_STD     — Gaussian timing jitter std [s] on acoustic reception
# #   MISS_PROB      — probability of dropping each acoustic measurement [0, 1]
# #   SOUND_SPEED    — acoustic propagation speed [m/s]
# #
# ACOUSTIC_DELAY = True
# JITTER_STD     = 0.1    # ±500 ms 1-sigma
# MISS_PROB      = 0.10    # 10 % dropout
# SOUND_SPEED    = 1500.0  # m/s
# # JITTER_STD     = 0.0
# # MISS_PROB      = 0.0
# SOUND_SPEED    = 1500.0


def run_simulations_s1(TRAJECTORY_TYPE, ACOUSTIC_DELAY, JITTER_STD, MISS_PROB, SOUND_SPEED, TDMA_FREQ, SAVE_DIR, ESKF_SIM, INIT_FROM_GT=False, INITIAL_RANGE_GUESS=10.0):

    imu_sim = ESKF_SIM.modelImuAsv
    _cv_sim = ESKF_SIM.modelCvRov
    gnss_sim = ESKF_SIM.sensorGnssAsv
    usbl_sim = ESKF_SIM.sensorUsbl
    range_sim = ESKF_SIM.sensorRange
    depth_sim = ESKF_SIM.sensorDepth

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
        rate_hz=TDMA_FREQ,
        acoustic_delay=ACOUSTIC_DELAY,
        jitter_std=JITTER_STD,
        miss_prob=MISS_PROB,
        sound_speed=SOUND_SPEED,
    )
    z_range_tseq = gen.generate_range(
        std_m=range_sim.range_std,
        lever_arm=range_sim.lever_arm,
        rate_hz=TDMA_FREQ,
        acoustic_delay=ACOUSTIC_DELAY,
        jitter_std=JITTER_STD,
        miss_prob=MISS_PROB,
        sound_speed=SOUND_SPEED,
    )
    z_depth_tseq = gen.generate_depth(
        std_m=depth_sim.depth_std,
        rate_hz=TDMA_FREQ,
        miss_prob=0.0,
    )

    if INIT_FROM_GT:
        x_init = _init_asv_from_gt(x_init_sim, asv_gt_tseq)
        x_init = _init_rov_from_gt(x_init, rov_gt_tseq)
    else:
        x_init = _init_asv_from_gnss(x_init_sim, z_gnss_tseq)
        x_init = _init_rov_from_usbl_range_depth(
            x_init=x_init,
            x_asv=x_init.nom.asv,
            z_usbl_tseq=z_usbl_tseq,
            z_range_tseq=z_range_tseq,
            z_depth_tseq=z_depth_tseq,
            range_guess=INITIAL_RANGE_GUESS,
        )

    traj_name = TRAJECTORY_TYPE.value

    # 3) Run scenarios
    print(f"[{traj_name}] Scenario 1: GNSS + Bearing only")
    upd_s1, pred_s1, zp_s1 = run_eskf_s1(
        eskf=eskf_sim,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        z_usbl_tseq=z_usbl_tseq,
    )

    # 4) Plot

    plotter_s1 = PlotterESKFJoint(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_gt_tseq,
        x_upds=upd_s1,
        x_preds=pred_s1,
        z_gnss_asv=z_gnss_tseq,
        z_usbl=z_usbl_tseq,
        z_preds=zp_s1,
        scenario_name=f"[{traj_name}] Scenario 1: Bearing-only",
        save_dir=f"{SAVE_DIR}/scenario1",
    )

    #plotter_s1.show()
    return plotter_s1




def run_simulations_s1_s2_s3(TRAJECTORY_TYPE, ACOUSTIC_DELAY, JITTER_STD, MISS_PROB, SOUND_SPEED, TDMA_FREQ, SAVE_DIR, ESKF_SIM, INIT_FROM_GT=False):

    imu_sim = ESKF_SIM.modelImuAsv
    _cv_sim = ESKF_SIM.modelCvRov
    gnss_sim = ESKF_SIM.sensorGnssAsv
    usbl_sim = ESKF_SIM.sensorUsbl
    range_sim = ESKF_SIM.sensorRange
    depth_sim = ESKF_SIM.sensorDepth

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
        rate_hz=TDMA_FREQ,
        acoustic_delay=ACOUSTIC_DELAY,
        jitter_std=JITTER_STD,
        miss_prob=MISS_PROB,
        sound_speed=SOUND_SPEED,
    )
    z_range_tseq = gen.generate_range(
        std_m=range_sim.range_std,
        lever_arm=range_sim.lever_arm,
        rate_hz=TDMA_FREQ,
        acoustic_delay=ACOUSTIC_DELAY,
        jitter_std=JITTER_STD,
        miss_prob=MISS_PROB,
        sound_speed=SOUND_SPEED,
    )
    z_depth_tseq = gen.generate_depth(
        std_m=depth_sim.depth_std,
        rate_hz=TDMA_FREQ,
        miss_prob=0.0,
    )

    if INIT_FROM_GT:
        x_init = _init_asv_from_gt(x_init_sim, asv_gt_tseq)
        x_init = _init_rov_from_gt(x_init, rov_gt_tseq)
    else:
        x_init = _init_asv_from_gnss(x_init_sim, z_gnss_tseq)
        x_init = _init_rov_from_usbl_range_depth(
            x_init=x_init,
            x_asv=x_init.nom.asv,
            z_usbl_tseq=z_usbl_tseq,
            z_range_tseq=z_range_tseq,
        z_depth_tseq=z_depth_tseq,
            range_guess=INITIAL_RANGE_SCALE,
        )

    traj_name = TRAJECTORY_TYPE.value
    # save_dir = f"results/{traj_name}_dt0_01_cv_0_02_aco_delay_{ACOUSTIC_DELAY}_jitter_{JITTER_STD}_miss_{MISS_PROB}_imu_driven_newStats"

    # 3) Run scenarios
    print(f"[{traj_name}] Scenario 1: GNSS + Bearing only")
    upd_s1, pred_s1, zp_s1 = run_eskf_s1(
        eskf=eskf_sim,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        z_usbl_tseq=z_usbl_tseq,
    )

    print(f"[{traj_name}] Scenario 2: GNSS + Bearing + Range")
    upd_s2, pred_s2, zp_s2 = run_eskf_s2(
        eskf=eskf_sim,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        z_usbl_tseq=z_usbl_tseq,
        z_range_tseq=z_range_tseq,
    )

    print(f"[{traj_name}] Scenario 3: GNSS + Bearing + Range + Depth")
    upd_s3, pred_s3, zp_s3 = run_eskf_s3(
        eskf=eskf_sim,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        z_usbl_tseq=z_usbl_tseq,
        z_range_tseq=z_range_tseq,
        z_depth_tseq=z_depth_tseq,
    )

    # 4) Plot
    scenario_rows = []

    plotter_s1 = PlotterESKFJoint(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_gt_tseq,
        x_upds=upd_s1,
        x_preds=pred_s1,
        z_gnss_asv=z_gnss_tseq,
        z_usbl=z_usbl_tseq,
        z_preds=zp_s1,
        scenario_name=f"[{traj_name}] Scenario 1: Bearing-only",
        save_dir=f"{SAVE_DIR}/scenario1",
    )
    scenario_rows.extend(plotter_s1.collect_statistics_rows())

    plotter_s2 = PlotterESKFJoint(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_gt_tseq,
        x_upds=upd_s2,
        x_preds=pred_s2,
        z_gnss_asv=z_gnss_tseq,
        z_usbl=z_usbl_tseq,
        z_range=z_range_tseq,
        z_preds=zp_s2,
        scenario_name=f"[{traj_name}] Scenario 2: Bearing + Range",
        save_dir=f"{SAVE_DIR}/scenario2",
    )
    scenario_rows.extend(plotter_s2.collect_statistics_rows())

    plotter_s3 = PlotterESKFJoint(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_gt_tseq,
        x_upds=upd_s3,
        x_preds=pred_s3,
        z_gnss_asv=z_gnss_tseq,
        z_usbl=z_usbl_tseq,
        z_range=z_range_tseq,
        z_depth=z_depth_tseq,
        z_preds=zp_s3,
        scenario_name=f"[{traj_name}] Scenario 3: Bearing + Range + Depth",
        save_dir=f"{SAVE_DIR}/scenario3",
    )
    scenario_rows.extend(plotter_s3.collect_statistics_rows())

    if scenario_rows:
        summary_path = Path(SAVE_DIR)
        summary_path.mkdir(parents=True, exist_ok=True)
        summary_file = summary_path / "eskf_statistics_summary.csv"
        fieldnames = []
        for row in scenario_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with summary_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(scenario_rows)

    plotter_s1.show()
    plotter_s2.show()
    plotter_s3.show()

def _init_asv_from_gt(
    x_init: JointEskfState,
    asv_gt_tseq,
) -> JointEskfState:
    asv_items = list(asv_gt_tseq.items())
    if not asv_items:
        return x_init

    _, asv0 = asv_items[0]
    asv_init = AsvNominalState(
        pos=asv0.pos,
        vel=asv0.vel,
        ori=asv0.ori,
        accm_bias=x_init.nom.asv.accm_bias,
        gyro_bias=x_init.nom.asv.gyro_bias,
    )
    return JointEskfState(
        nom=JointNominalState(asv=asv_init, rov=x_init.nom.rov),
        err=x_init.err,
    )

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
        pos=WithXYZ.from_array(pos1),
        vel=WithXYZ.from_array(np.zeros(3)),
        ori=RotationQuaterion.from_euler(np.asarray([0.0, 0.0, yaw], dtype=float)),
        accm_bias=asv_prev.accm_bias,
        gyro_bias=asv_prev.gyro_bias,
    )
    return JointEskfState(
        nom=JointNominalState(asv=asv_init, rov=x_init.nom.rov),
        err=x_init.err,
    )

def _init_rov_from_usbl_range_depth(
    x_init: JointEskfState, # Initial guess (can be from ASV init or just default)
    x_asv: AsvNominalState, # ASV state (after init from GNSS or in case we init runtime)
    z_usbl_tseq,
    z_range_tseq,
    z_depth_tseq,
    range_guess=10.0,
    usbl_lever_arm=np.array([0.0, 0.0, 1.2]),
) -> JointEskfState:
    usbl_items = list(z_usbl_tseq.items())
    if not usbl_items:
        return x_init

    timestamp, usbl0 = usbl_items[0]
    # _range0 = z_range_tseq.get_t(timestamp) if z_range_tseq else None
    # _depth0 = z_depth_tseq.get_t(timestamp) if z_depth_tseq else None
    # range = float(_range0) if _range0 else None
    # depth = float(_depth0) if _depth0 else None

    az = usbl0[0]  # azimuth
    el = usbl0[1]  # elevation

    sensor_pos = x_asv.pos + x_asv.ori.as_rotmat() @ usbl_lever_arm  # assuming lever arm is zero for simplicity

    # scenario 1, default
    rov_pos = WithXYZ.from_array([
                sensor_pos.x + range_guess * np.cos(az) * np.cos(el),
                sensor_pos.y + range_guess * np.sin(az) * np.cos(el),
                sensor_pos.z + range_guess * np.sin(el),
            ])    
    rov_vel = x_init.nom.rov.vel  # default to initial guess


    if z_range_tseq is not None:
        _t, range = z_range_tseq.get_idx(0)
        range = float(range)
        # 2D position from range + bearing, keep z from initial guess
        rov_pos = WithXYZ.from_array([
            sensor_pos.x + range * np.cos(az) * np.cos(el),
            sensor_pos.y + range * np.sin(az) * np.cos(el),
            sensor_pos.z + range * np.sin(el),  # this will be zero if elevation is zero, otherwise we don't have better info than the initial guess
        ])
    if z_depth_tseq is not None and z_range_tseq is not None:
        _t, range = z_range_tseq.get_idx(0)
        _t, depth = z_depth_tseq.get_idx(0)
        range = float(range)
        depth = float(depth)
        rov_pos = WithXYZ.from_array([
            sensor_pos.x + range * np.cos(az) * np.cos(el),
            sensor_pos.y + range * np.sin(az) * np.cos(el),
            depth,
        ])

    rov_init = RovNominalCV(
        pos=rov_pos,
        vel=rov_vel,
    )
    return JointEskfState(
        nom=JointNominalState(asv=x_asv, rov=rov_init),
        err=x_init.err,
    )

def _init_rov_from_gt(
    x_init: JointEskfState,
    rov_gt_tseq,
) -> JointEskfState:
    rov_items = list(rov_gt_tseq.items())
    if not rov_items:
        return x_init

    _, rov0 = rov_items[0]
    rov_init = RovNominalCV(
        pos=rov0.pos,
        vel=rov0.vel,
    )
    rov_err_init_std_sim = np.repeat(
        repeats=3,
        a=[
            2.0,    # pos
            0.2,    # vel
        ],
    )
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
    err_init = MultiVarGauss[JointErrorState](
        JointErrorState.from_array(np.zeros(JointIdx.N)),
        np.diag(np.concatenate((asv_err_init_std_sim, rov_err_init_std_sim)) ** 2),
    )
    return JointEskfState(
        nom=JointNominalState(asv=x_init.nom.asv, rov=rov_init),
        err=err_init,
    )