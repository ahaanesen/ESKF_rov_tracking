import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from asv_states import ASVState, RangeMeasurement, UsblMeasurement
from eskf import ESKF_cv, ESKF_imu
from rov_states import DepthMeasurement, EskfState, ImuMeasurement, NominalState
from senfuslib import MultiVarGauss, TimeSequence
from plotting import PlotterESKF

from tuning_sim import (eskf_sim, rov_est_init_sim,
                        start_time_sim, end_time_sim,
                        usbl_sim, range_sim, depth_sim, imu_sim)
from config import RUN
from utils.generate_trajectory import generate_trajectories
from utils.generate_measurements import MeasurementGenerator
from run_scenarios import run_eskf_s1, run_eskf_s2, run_eskf_s3, run_eskf_s4


# def run_eskf(eskf: ESKF,
#              x_est_init: EskfState,
#              z_imu_tseq: TimeSequence[ImuMeasurement],
#              z_gnss_tseq: TimeSequence[GnssMeasurement],
#              ) -> tuple[TimeSequence[EskfState],
#                         TimeSequence[EskfState],
#                         TimeSequence[MultiVarGauss[ImuMeasurement]]]:

#     t_prev = z_imu_tseq.times[0]
#     x_est_prev = x_est_init
#     x_est_pred_tseq = TimeSequence([(t_prev, x_est_init)])
#     z_est_pred_tseq = TimeSequence()
#     x_est_upd_tseq = TimeSequence()

#     gnss_copy = z_gnss_tseq.copy()
#     for t_imu, z_imu in tqdm(z_imu_tseq.items()):

#         # Handle gnss measurements that has arrived since last imu measurement
#         while gnss_copy and gnss_copy.times[0] <= t_imu:
#             t_gps, z_gnss = gnss_copy.pop_idx(0)
#             dt = t_gps - t_prev
#             x_est_pred = eskf.predict_from_imu(x_est_prev, z_imu, dt)
#             x_est_upd, z_est_pred = eskf.update_from_gnss(x_est_pred, z_gnss)
#             x_est_prev = x_est_upd
#             x_est_pred_tseq.insert(t_gps, x_est_pred)
#             z_est_pred_tseq.insert(t_gps, z_est_pred)
#             x_est_upd_tseq.insert(t_gps, x_est_upd)
#             t_prev = t_gps

#         dt = t_imu - t_prev
#         if dt > 0:
#             x_est_pred = eskf.predict_from_imu(x_est_prev, z_imu, dt)
#             x_est_pred_tseq.insert(t_imu, x_est_pred)
#             x_est_prev = x_est_pred
#         t_prev = t_imu

#     return x_est_upd_tseq, x_est_pred_tseq, z_est_pred_tseq


def main():
    # if RUN == 'sim':
    #     fname = fname_data_sim
    #     esfk = eskf_sim
    #     x_est_init = x_est_init_sim
    #     tslice = (start_time_sim, end_time_sim, imu_min_dt_sim)
    #     gnss_min_dt = gnss_min_dt_sim

    # elif RUN == 'real':
    #     fname = fname_data_real
    #     esfk = eskf_real
    #     x_est_init = x_est_init_real
    #     tslice = (start_time_real, end_time_real, imu_min_dt_real)
    #     gnss_min_dt = gnss_min_dt_real

    # x_gt, z_imu_tseq, z_gnss_tseq = load_data(fname)

    # z_imu_tseq = z_imu_tseq.slice_time(*tslice)
    # z_gnss_tseq = z_gnss_tseq.slice_time(z_imu_tseq.times[0],
    #                                      z_imu_tseq.times[-1],
    #                                      gnss_min_dt)

    # out = run_eskf(esfk, x_est_init, z_imu_tseq, z_gnss_tseq)
    # x_est_upd_tseq, x_est_pred_tseq, z_est_pred_tseq = out

    # PlotterESKF(x_gts=x_gt,
    #             z_imu=z_imu_tseq,
    #             z_gnss=z_gnss_tseq,
    #             x_preds=x_est_pred_tseq,
    #             z_preds=z_est_pred_tseq,
    #             x_upds=x_est_upd_tseq,
    #             ).show()
    # plt.show(block=True)
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
        rate_hz=5.0,
    )
    # z_imu_tseq = gen.generate_imu(
    #     accm_std=imu_sim.accm_std,
    #     gyro_std=imu_sim.gyro_std,
    #     rate_hz=100.0,
    # )

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
    # PlotterESKF(
    #     x_gts=rov_gt_tseq,
    #     x_upds=upd_s3,
    #     x_preds=pred_s3,
    # ).show()
    # plt.show(block=True)

if __name__ == '__main__':
    main()
