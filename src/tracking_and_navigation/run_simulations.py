from tracking_and_navigation.generate_trajectories import generate_trajectories
from tracking_and_navigation.generate_measurements import MeasurementGenerator

from tracking_and_navigation.run_eskf import run_eskf_s1, run_eskf_s2, run_eskf_s3
from tracking_and_navigation.plotting import PlotterESKFJoint

from tracking_and_navigation.tuning_sim import (
    eskf_sim,
    x_init_sim,   # recommend renaming from rov_est_init_sim -> x_init_sim in this package
    usbl_sim,
    range_sim,
    depth_sim,
    gnss_sim,     # recommend adding to tuning_sim: SensorGNSS_ASV config
    imu_sim,      # recommend adding to tuning_sim: IMU noise params for measurement generation
)

def run_simulations():

    # 1) Ground truth
    asv_gt_tseq, rov_gt_tseq, _ = generate_trajectories(duration=300, dt=0.1)

    # 2) Measurements (ASV IMU + ASV GNSS + tracking sensors)
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
    print("ASV GT down min/max:", min(v.pos[2] for v in asv_gt_tseq.values), max(v.pos[2] for v in asv_gt_tseq.values))
    print("GNSS down min/max:", min(v.pos[2] for v in z_gnss_tseq.values), max(v.pos[2] for v in z_gnss_tseq.values))
    print("GNSS first sample:", z_gnss_tseq.values[0].pos)

    z_usbl_tseq = gen.generate_usbl(std_rad=usbl_sim.usbl_std, lever_arm=usbl_sim.lever_arm, rate_hz=1.0)
    z_range_tseq = gen.generate_range(std_m=range_sim.range_std, lever_arm=range_sim.lever_arm, rate_hz=1.0)
    z_depth_tseq = gen.generate_depth(std_m=depth_sim.depth_std, rate_hz=1.0)

    # 3) Run scenarios (each scenario includes GNSS in the measurement list)
    # print("Running tracking_and_navigation Scenario 1: GNSS + USBL")
    # upd_s1, pred_s1 = run_eskf_s1(
    #     eskf=eskf_sim,
    #     x_init=x_init_sim,
    #     z_imu_tseq=z_imu_tseq,
    #     z_gnss_tseq=z_gnss_tseq,
    #     z_usbl_tseq=z_usbl_tseq,
    # )

    # print("Running tracking_and_navigation Scenario 2: GNSS + USBL + Range")
    # upd_s2, pred_s2 = run_eskf_s2(
    #     eskf=eskf_sim,
    #     x_init=x_init_sim,
    #     z_imu_tseq=z_imu_tseq,
    #     z_gnss_tseq=z_gnss_tseq,
    #     z_usbl_tseq=z_usbl_tseq,
    #     z_range_tseq=z_range_tseq,
    # )

    print("Running tracking_and_navigation Scenario 3: GNSS + USBL + Range + Depth")
    upd_s3, pred_s3 = run_eskf_s3(
        eskf=eskf_sim,
        x_init=x_init_sim,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        z_usbl_tseq=z_usbl_tseq,
        z_range_tseq=z_range_tseq,
        z_depth_tseq=z_depth_tseq,
    )

    # 4) Plot (joint plotter)
    PlotterESKFJoint(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_gt_tseq,
        x_upds=upd_s3,
        x_preds=pred_s3,
        z_gnss_asv=z_gnss_tseq,
        z_usbl=z_usbl_tseq,
        z_range=z_range_tseq,
        z_depth=z_depth_tseq,
        scenario_name="tracking_and_navigation — Scenario 3 (Joint)",
        save_dir="plots/tracking_and_navigation/scenario3",
    ).show()