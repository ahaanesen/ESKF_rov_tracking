import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from asv_states import ASVState, RangeMeasurement, UsblMeasurement
from eskf import ESKF_cv, ESKF_imu

from rov_states import DepthMeasurement, EskfState, ImuMeasurement
from senfuslib import MultiVarGauss, TimeSequence

# Only running ESKF on simulated data


def _run_cv_scenario(
    eskf: ESKF_cv,
    rov_est_init: EskfState,
    asv_state_tseq: TimeSequence[ASVState],
    measurements,
    t_start: float,
    desc: str,
    include_init_in_upd: bool,
    include_init_in_pred: bool,
) -> tuple[TimeSequence[EskfState], TimeSequence[EskfState]]:
    t_prev = t_start
    rov_est_prev = rov_est_init

    rov_upd_tseq = TimeSequence([(t_prev, rov_est_init)]) if include_init_in_upd else TimeSequence()
    rov_pred_tseq = TimeSequence([(t_prev, rov_est_init)]) if include_init_in_pred else TimeSequence()

    for t, z in tqdm(measurements, desc=desc):
        dt = t - t_prev
        if dt < 0:
            continue

        rov_est_pred = eskf.predict_with_cv(rov_est_prev, dt)

        if isinstance(z, UsblMeasurement):
            asv_state = asv_state_tseq.at_time(t)
            rov_est_upd, _ = eskf.update_from_usbl(rov_est_pred, asv_state, z)
        elif isinstance(z, RangeMeasurement):
            asv_state = asv_state_tseq.at_time(t)
            rov_est_upd, _ = eskf.update_from_range(rov_est_pred, asv_state, z)
        elif isinstance(z, DepthMeasurement):
            rov_est_upd, _ = eskf.update_from_depth(rov_est_pred, z)
        else:
            raise TypeError(f"Unsupported measurement type: {type(z)}")

        # rov_upd_tseq.insert(t, rov_est_upd)
        # rov_pred_tseq.insert(t, rov_est_pred)
        if t not in rov_upd_tseq:
            rov_upd_tseq.insert(t, rov_est_upd)
        if t not in rov_pred_tseq:
            rov_pred_tseq.insert(t, rov_est_pred)
        rov_est_prev = rov_est_upd
        t_prev = t

    return rov_upd_tseq, rov_pred_tseq

# Scenario 1: Bearing only (USBL: azi+elev), CV model for ROV, ASV with known trajectory
def run_eskf_s1(eskf: ESKF_cv,
                rov_est_init: EskfState,
                asv_state_tseq: TimeSequence[ASVState],
                z_usbl_tseq: TimeSequence[UsblMeasurement],
                ) -> tuple[TimeSequence[EskfState],
                           TimeSequence[EskfState]]:
    return _run_cv_scenario(
        eskf=eskf,
        rov_est_init=rov_est_init,
        asv_state_tseq=asv_state_tseq,
        measurements=z_usbl_tseq.items(),
        t_start=z_usbl_tseq.times[0],
        desc="Scenario 1",
        include_init_in_upd=False,
        include_init_in_pred=False,
    )


# Scenario 2: Bearing+range, CV model for ROV, ASV with known trajectory
def run_eskf_s2(eskf: ESKF_cv,
                rov_est_init: EskfState,
                asv_state_tseq: TimeSequence[ASVState],
                z_usbl_tseq: TimeSequence[UsblMeasurement],
                z_range_tseq: TimeSequence[RangeMeasurement],
                ) -> tuple[TimeSequence[EskfState], TimeSequence[EskfState]]:
    return _run_cv_scenario(
        eskf=eskf,
        rov_est_init=rov_est_init,
        asv_state_tseq=asv_state_tseq,
        measurements=z_usbl_tseq.combine_with(z_range_tseq),
        t_start=z_usbl_tseq.times[0],
        desc="Scenario 2",
        include_init_in_upd=True,
        include_init_in_pred=False,
    )


# Scenario 3: Bearing+range+depth, CV model for ROV, ASV with known trajectory
def run_eskf_s3(eskf: ESKF_cv,
                rov_est_init: EskfState,
                asv_state_tseq: TimeSequence[ASVState],
                z_usbl_tseq: TimeSequence[UsblMeasurement],
                z_range_tseq: TimeSequence[RangeMeasurement],
                z_depth_tseq: TimeSequence[DepthMeasurement],
                ) -> tuple[TimeSequence[EskfState], TimeSequence[EskfState]]:
    return _run_cv_scenario(
        eskf=eskf,
        rov_est_init=rov_est_init,
        asv_state_tseq=asv_state_tseq,
        measurements=z_usbl_tseq.combine_with(z_range_tseq, z_depth_tseq),
        t_start=min(z_usbl_tseq.t_min, z_range_tseq.t_min, z_depth_tseq.t_min),
        desc="Scenario 3",
        include_init_in_upd=True,
        include_init_in_pred=False,
    )


# Scenario 4: Bearing+range+depth, IMU model for ROV, ASV with known trajectory
def run_eskf_s4(eskf: ESKF_imu,
                rov_est_init: EskfState,
                asv_state_tseq: TimeSequence[ASVState],
                z_imu_tseq: TimeSequence[ImuMeasurement],
                z_usbl_tseq: TimeSequence[UsblMeasurement],
                z_range_tseq: TimeSequence[RangeMeasurement],
                z_depth_tseq: TimeSequence[DepthMeasurement],
                ) -> tuple[TimeSequence[EskfState], TimeSequence[EskfState]]:
    
    t_prev = z_imu_tseq.times[0]
    rov_est_prev = rov_est_init
    rov_upd_tseq = TimeSequence([(t_prev, rov_est_init)])
    rov_pred_tseq = TimeSequence()

    # Create a copy for the measurement pool
    measurements = z_usbl_tseq.combine_with(z_range_tseq, z_depth_tseq)
    
    # Iterate primarily over high-frequency IMU
    for t_imu, z_imu in tqdm(z_imu_tseq.items(), desc="Scenario 4 (IMU)"):
        
        # Check if any "slow" measurements arrived since last IMU step
        while measurements and measurements.peek()[0] <= t_imu:
            t_m, z_m = next(measurements)
            dt = t_m - t_prev
            
            # Predict up to the measurement time
            rov_est_pred = eskf.predict_from_imu(rov_est_prev, z_imu, dt)
            rov_pred_tseq.insert(t_m, rov_est_pred) 
            
            # Update based on type
            if isinstance(z_m, UsblMeasurement):
                rov_est_upd, _ = eskf.update_from_usbl(rov_est_pred, asv_state_tseq.at_time(t_m), z_m)
            elif isinstance(z_m, RangeMeasurement):
                rov_est_upd, _ = eskf.update_from_range(rov_est_pred, asv_state_tseq.at_time(t_m), z_m)
            elif isinstance(z_m, DepthMeasurement):
                rov_est_upd, _ = eskf.update_from_depth(rov_est_pred, z_m)
            
            rov_est_prev = rov_est_upd
            t_prev = t_m
            rov_upd_tseq.insert(t_m, rov_est_upd)

        # Standard IMU Prediction step
        dt = t_imu - t_prev
        if dt > 0:
            rov_est_pred = eskf.predict_from_imu(rov_est_prev, z_imu, dt)
            rov_pred_tseq.insert(t_imu, rov_est_pred)
            rov_est_prev = rov_est_pred
            t_prev = t_imu

    return rov_upd_tseq, rov_pred_tseq
