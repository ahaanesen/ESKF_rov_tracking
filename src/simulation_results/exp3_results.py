# scripts/exp3_results.py

from __future__ import annotations

from pathlib import Path
import sys

# Ensure workspace src/ is on sys.path when running this file directly.
WORKSPACE_SRC = Path(__file__).resolve().parents[1]
if str(WORKSPACE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_SRC))


import numpy as np

from tracking_and_navigation.generate_trajectories import TrajectoryType, generate_trajectories
from tracking_and_navigation.tuning_sim import eskf_sim, x_init_sim
from tracking_and_navigation.run_simulations import (
    _init_asv_from_gnss,
    _init_asv_from_gt,
    _init_rov_from_gt,
    _init_rov_from_usbl_range_depth,
)
from tracking_and_navigation.generate_measurements import MeasurementGenerator
from tracking_and_navigation.plotting import PlotterESKFJoint
from tracking_and_navigation.run_eskf import run_eskf_s1, run_eskf_s2, run_eskf_s3


from simulation_results.eskf_experiments import (  # rename to your actual module file
    extract_eskf_result,
    generate_tdma_figures,
    results_to_dataframe,
    save_results_csv,
    generate_tdma_figures,
    save_figures,
)

TRAJECTORY_TYPE = TrajectoryType.LINEAR_TURNS

ACOUSTIC_DELAY = True
JITTER_STD     = 0.0
MISS_PROB      = 0.0
SOUND_SPEED    = 1500.0
#TDMA_FREQ      = 0.2  # 1 msg / 5 s

# TRUE_RANGE = np.sqrt(269)  # pre-computed true range for this scenario
TRUE_RANGE = 223.75 # for linear turns, pre-computed true range is slightly different due to different trajectory geometry

TDMA_SLOT_LENGTHS = [5.0, 10.0, 20.0, 30.0, 60.0, 120.0]

# TDMA_SLOT_LENGTHS = [5.0, 10.0, 20.0]

def main():
    outdir = Path(f"results/{TRAJECTORY_TYPE.value}/exp3_tdma_new")
    outdir.mkdir(parents=True, exist_ok=True)

    results = []

    traj_name = TRAJECTORY_TYPE.value

    imu_sim = eskf_sim.modelImuAsv
    gnss_sim = eskf_sim.sensorGnssAsv
    usbl_sim = eskf_sim.sensorUsbl
    range_sim = eskf_sim.sensorRange
    depth_sim = eskf_sim.sensorDepth

    asv_gt_tseq, rov_gt_tseq, _ = generate_trajectories(
        duration=300,
        dt=0.01,
        trajectory_type=TRAJECTORY_TYPE,
    )

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
    packet_mask = gen.generate_acoustic_packet_mask(
        rate_hz=0.2,
        miss_prob=0.0,
        loss_seed=42,
    )

    for tdma_slot_length in TDMA_SLOT_LENGTHS:
        z_usbl_tseq = gen.generate_usbl(
            std_rad=usbl_sim.usbl_std,
            lever_arm=usbl_sim.lever_arm,
            rate_hz=1/tdma_slot_length,
            acoustic_delay=ACOUSTIC_DELAY,
            jitter_std=JITTER_STD,
            packet_mask=packet_mask,
            sound_speed=SOUND_SPEED,
        )
        z_range_tseq = gen.generate_range(
            std_m=range_sim.range_std,
            lever_arm=range_sim.lever_arm,
            rate_hz=1/tdma_slot_length,
            acoustic_delay=ACOUSTIC_DELAY,
            jitter_std=JITTER_STD,
            packet_mask=packet_mask,
            sound_speed=SOUND_SPEED,
        )
        z_depth_tseq = gen.generate_depth(
            std_m=depth_sim.depth_std,
            rate_hz=1/tdma_slot_length,
            packet_mask=packet_mask,
        )

        print(f"[exp3] Running tdma slot length = {tdma_slot_length:.3f}")

        save_dir = outdir / f"raw/tdma_slot_length_{tdma_slot_length:.3f}"
        save_dir.mkdir(parents=True, exist_ok=True)

        scenario_rows = []
        for scenario in [1,2,3]:
            plotter = run_simulation(
                SCENARIO=scenario,
                TRAJ_NAME=traj_name,
                ACOUSTIC_DELAY=ACOUSTIC_DELAY,
                JITTER_STD=JITTER_STD,
                MISS_PROB=MISS_PROB,
                SOUND_SPEED=SOUND_SPEED,
                TDMA_FREQ=1.0 / tdma_slot_length,
                SAVE_DIR=str(save_dir / f"scenario_{scenario}"),
                ESKF_SIM=eskf_sim,
                INIT_FROM_GT=False,
                INITIAL_RANGE_GUESS= TRUE_RANGE,
                asv_gt_tseq=asv_gt_tseq,
                rov_gt_tseq=rov_gt_tseq,
                z_imu_tseq=z_imu_tseq,
                z_gnss_tseq=z_gnss_tseq,
                z_usbl_tseq=z_usbl_tseq,
                z_range_tseq=z_range_tseq,
                z_depth_tseq=z_depth_tseq,
            )
            plotter.show()
            res = extract_eskf_result(
                plotter,
                experiment="exp3",
                scenario=f"scenario{scenario}",
                estimator="ESKF",
                sigma_a=eskf_sim.modelCvRov.sigma_a,
                tdma_interval=tdma_slot_length,
                init_range_scale=1.0,
                run_idx=0,
                divergence_threshold=10.0,
            )
            results.extend(res)
            # scenario_rows.extend(plotter.collect_statistics_rows())


        # if scenario_rows:
        #     summary_path = Path(SAVE_DIR)
        #     summary_path.mkdir(parents=True, exist_ok=True)
        #     summary_file = summary_path / "eskf_statistics_summary.csv"
        #     fieldnames = []
        #     for row in scenario_rows:
        #         for key in row.keys():
        #             if key not in fieldnames:
        #                 fieldnames.append(key)

        #     with summary_file.open("w", newline="", encoding="utf-8") as f:
        #         writer = csv.DictWriter(f, fieldnames=fieldnames)
        #         writer.writeheader()
        #         writer.writerows(scenario_rows)
            

    df = results_to_dataframe(results)
    save_results_csv(df, str(outdir / "exp3_tdma_slot_length_results.csv"))

    figs = generate_tdma_figures(df)
    save_figures(figs, str(outdir / "figures"))

    print(f"Wrote {outdir / 'exp3_tdma_slot_length_results.csv'}")
    print(f"Wrote figures to {outdir / 'figures'}")


def run_simulation(
    SCENARIO: int,
    TRAJ_NAME: str,
    ACOUSTIC_DELAY: bool,
    JITTER_STD: float,
    MISS_PROB: float,
    SOUND_SPEED: float,
    TDMA_FREQ: float,
    SAVE_DIR: str,
    ESKF_SIM,
    INIT_FROM_GT: bool,
    INITIAL_RANGE_GUESS: float,
    asv_gt_tseq,
    rov_gt_tseq,
    z_imu_tseq,
    z_gnss_tseq,
    z_usbl_tseq,
    z_range_tseq,
    z_depth_tseq,
):
    if SCENARIO == 1:
        if INIT_FROM_GT:
            x_init = _init_asv_from_gt(x_init_sim, asv_gt_tseq)
            x_init = _init_rov_from_gt(x_init, rov_gt_tseq)
        else:
            x_init = _init_asv_from_gnss(x_init_sim, z_gnss_tseq)
            x_init = _init_rov_from_usbl_range_depth(
                x_init=x_init,
                x_asv=x_init.nom.asv,
                z_usbl_tseq=z_usbl_tseq,
                z_range_tseq=None,
                z_depth_tseq=None,
                range_guess=INITIAL_RANGE_GUESS,
            )
        
        print(f"[{TRAJ_NAME}] Scenario 1: GNSS + Bearing only")
        upd_s1, pred_s1, zp_s1 = run_eskf_s1(
            eskf=eskf_sim,
            x_init=x_init,
            z_imu_tseq=z_imu_tseq,
            z_gnss_tseq=z_gnss_tseq,
            z_usbl_tseq=z_usbl_tseq,
        )

        plotter_s1 = PlotterESKFJoint(
            rov_gt=rov_gt_tseq,
            asv_gt=asv_gt_tseq,
            x_upds=upd_s1,
            x_preds=pred_s1,
            z_gnss_asv=z_gnss_tseq,
            z_usbl=z_usbl_tseq,
            z_preds=zp_s1,
            scenario_name=f"[{TRAJ_NAME}] Scenario 1: Bearing-only",
            save_dir=f"{SAVE_DIR}/scenario1",
        )

        return plotter_s1

    if SCENARIO == 2:
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
                z_depth_tseq=None,
                range_guess=INITIAL_RANGE_GUESS,
            )

        print(f"[{TRAJ_NAME}] Scenario 2: GNSS + Bearing + Range")
        upd_s2, pred_s2, zp_s2 = run_eskf_s2(
            eskf=eskf_sim,
            x_init=x_init,
            z_imu_tseq=z_imu_tseq,
            z_gnss_tseq=z_gnss_tseq,
            z_usbl_tseq=z_usbl_tseq,
            z_range_tseq=z_range_tseq,
        )

        plotter_s2 = PlotterESKFJoint(
            rov_gt=rov_gt_tseq,
            asv_gt=asv_gt_tseq,
            x_upds=upd_s2,
            x_preds=pred_s2,
            z_gnss_asv=z_gnss_tseq,
            z_usbl=z_usbl_tseq,
            z_range=z_range_tseq,
            z_preds=zp_s2,
            scenario_name=f"[{TRAJ_NAME}] Scenario 2: Bearing + Range",
            save_dir=f"{SAVE_DIR}/scenario2",
        )
        return plotter_s2


    if SCENARIO == 3:
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
        print(f"[{TRAJ_NAME}] Scenario 3: GNSS + Bearing + Range + Depth")
        upd_s3, pred_s3, zp_s3 = run_eskf_s3(
            eskf=eskf_sim,
            x_init=x_init,
            z_imu_tseq=z_imu_tseq,
            z_gnss_tseq=z_gnss_tseq,
            z_usbl_tseq=z_usbl_tseq,
            z_range_tseq=z_range_tseq,
            z_depth_tseq=z_depth_tseq,
        )
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
            scenario_name=f"[{TRAJ_NAME}] Scenario 3: Bearing + Range + Depth",
            save_dir=f"{SAVE_DIR}/scenario3",
        )
        return plotter_s3

if __name__ == "__main__":
    main()