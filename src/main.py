import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from tracking_only.asv_states import ASVState, RangeMeasurement, UsblMeasurement
from tracking_only.eskf import ESKF_cv, ESKF_imu
from tracking_only.rov_states import DepthMeasurement, EskfState, ImuMeasurement, NominalState
from senfuslib import MultiVarGauss, TimeSequence
from tracking_only.plotting import PlotterESKF

from tracking_only.tuning_sim import (eskf_sim, rov_est_init_sim,
                        start_time_sim, end_time_sim,
                        usbl_sim, range_sim, depth_sim)
from config import RUN
from utils.generate_trajectory import generate_trajectories
from utils.generate_measurements import MeasurementGenerator
from tracking_only.run_scenarios import run_eskf_s1, run_eskf_s2, run_eskf_s3


def main():
    # if RUN == 'tracking_only':


    # elif RUN == 'tracking_and_navigation':

    # 1. Ground truth trajectories
    asv_tseq, rov_gt_tseq = generate_trajectories(duration=300, dt=0.1)

    # 2. Simulate sensor measurements from ground truth
    gen = MeasurementGenerator(asv_tseq, rov_gt_tseq)

    z_usbl_tseq  = gen.generate_usbl(
        std_rad=usbl_sim.usbl_std,
        lever_arm=usbl_sim.lever_arm,
        rate_hz=1.0,
    )
    z_range_tseq = gen.generate_range(
        std_m=range_sim.range_std,
        lever_arm=range_sim.lever_arm,
        rate_hz=1.0,
    )
    z_depth_tseq = gen.generate_depth(
        std_m=depth_sim.depth_std,
        rate_hz=1.0,
    )

    # 3. Run scenarios — comment/uncomment as needed
    print("Running Scenario 1: USBL bearing only, CV model")
    upd_s1, pred_s1 = run_eskf_s1(
        eskf=eskf_sim,
        rov_est_init=rov_est_init_sim,
        asv_state_tseq=asv_tseq,
        z_usbl_tseq=z_usbl_tseq,
    )

    print("Running Scenario 2: USBL bearing + range, CV model")
    upd_s2, pred_s2 = run_eskf_s2(
        eskf=eskf_sim,
        rov_est_init=rov_est_init_sim,
        asv_state_tseq=asv_tseq,
        z_usbl_tseq=z_usbl_tseq,
        z_range_tseq=z_range_tseq,
    )

    print("Running Scenario 3: USBL bearing + range + depth, CV model")
    upd_s3, pred_s3 = run_eskf_s3(
        eskf=eskf_sim,
        rov_est_init=rov_est_init_sim,
        asv_state_tseq=asv_tseq,
        z_usbl_tseq=z_usbl_tseq,
        z_range_tseq=z_range_tseq,
        z_depth_tseq=z_depth_tseq,
    )

    # 4. Plot — adapt to your PlotterESKF interface
    PlotterESKF(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_tseq,
        rov_upds=upd_s1,
        rov_preds=pred_s1,
        z_usbl=z_usbl_tseq,
        scenario_name="Scenario 1: Bearing only (CV)",
        save_dir="plots/scenario1",
    ).show()
    PlotterESKF(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_tseq,
        rov_upds=upd_s2,
        rov_preds=pred_s2,
        z_usbl=z_usbl_tseq,
        z_range=z_range_tseq,
        scenario_name="Scenario 2: Bearing + Range (CV)",
        save_dir="plots/scenario2",
    ).show()
    PlotterESKF(
        rov_gt=rov_gt_tseq,
        asv_gt=asv_tseq,
        rov_upds=upd_s3,
        rov_preds=pred_s3,
        z_usbl=z_usbl_tseq,
        z_range=z_range_tseq,
        z_depth=z_depth_tseq,
        scenario_name="Scenario 3: Bearing + Range + Depth (CV)",
        save_dir="plots/scenario3",
    ).show()

if __name__ == '__main__':
    main()
