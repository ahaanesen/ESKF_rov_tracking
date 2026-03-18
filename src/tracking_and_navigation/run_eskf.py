import numpy as np
from tqdm import tqdm

from senfuslib import TimeSequence

from tracking_and_navigation.states import JointEskfState
from tracking_and_navigation.eskf import ESKF_joint

from tracking_and_navigation.measurements import (
    GnssMeasurement,
    ImuMeasurement,
    UsblMeasurement,
    RangeMeasurement,
    DepthMeasurement,
)


def _merge_measurements(*tseqs: TimeSequence):
    """
    Merge multiple TimeSequence measurement streams into a sorted python list [(t, z), ...].
    """
    all_meas = []
    for ts in tseqs:
        if ts is None:
            continue
        all_meas += [(t, z) for t, z in ts.items()]
    all_meas.sort(key=lambda x: x[0])
    return all_meas


def _run_joint_scenario(
    eskf: ESKF_joint,
    x_init: JointEskfState,
    z_imu_tseq: TimeSequence[ImuMeasurement],
    meas_list,  # list[(t, measurement)]
    desc: str,
    include_init_in_upd: bool,
    include_init_in_pred: bool,
) -> tuple[TimeSequence[JointEskfState], TimeSequence[JointEskfState]]:
    """
    Run joint filter driven by IMU, with async updates (USBL/range/depth).

    Strategy:
      - iterate over IMU times
      - between IMU steps, apply any pending low-rate measurements whose timestamp <= current IMU time
      - for each measurement, first predict from last time -> meas time using IMU at current index (approx)
    """
    if not z_imu_tseq.times:
        raise ValueError("z_imu_tseq is empty")

    # Measurement cursor
    m_idx = 0
    m_len = len(meas_list)

    t_prev = z_imu_tseq.times[0]
    x_prev = x_init

    upd_tseq = TimeSequence([(t_prev, x_init)]) if include_init_in_upd else TimeSequence()
    pred_tseq = TimeSequence([(t_prev, x_init)]) if include_init_in_pred else TimeSequence()

    for t_imu, z_imu in tqdm(z_imu_tseq.items(), desc=desc):
        # 1) Handle any low-rate measurements that arrived up to this IMU time
        while m_idx < m_len and meas_list[m_idx][0] <= t_imu:
            t_m, z_m = meas_list[m_idx]
            dt_m = t_m - t_prev
            if dt_m < 0:
                m_idx += 1
                continue

            # Predict to measurement time using the latest available IMU sample (z_imu)
            x_pred = eskf.predict_from_imu(x_prev, z_imu, dt_m)
            if t_m not in pred_tseq:
                pred_tseq.insert(t_m, x_pred)

            # Update by type
            if isinstance(z_m, GnssMeasurement):
                x_upd, _ = eskf.update_from_gnss_asv(x_pred, z_m)
            elif isinstance(z_m, UsblMeasurement):
                x_upd, _ = eskf.update_from_usbl(x_pred, z_m)
            elif isinstance(z_m, RangeMeasurement):
                x_upd, _ = eskf.update_from_range(x_pred, z_m)
            elif isinstance(z_m, DepthMeasurement):
                x_upd, _ = eskf.update_from_depth(x_pred, z_m)
            else:
                raise TypeError(f"Unsupported measurement type: {type(z_m)}")

            if t_m not in upd_tseq:
                upd_tseq.insert(t_m, x_upd)

            x_prev = x_upd
            t_prev = t_m
            m_idx += 1

        # 2) Regular IMU propagation to the IMU timestamp
        dt = t_imu - t_prev
        if dt > 0:
            x_pred = eskf.predict_from_imu(x_prev, z_imu, dt)
            if t_imu not in pred_tseq:
                pred_tseq.insert(t_imu, x_pred)
            x_prev = x_pred
            t_prev = t_imu

    return upd_tseq, pred_tseq


# -----------------------------------------------------------------------------
# Scenario 1: USBL bearing only
# -----------------------------------------------------------------------------
def run_eskf_s1(
    eskf: ESKF_joint,
    x_init: JointEskfState,
    z_imu_tseq: TimeSequence[ImuMeasurement],
    z_gnss_tseq: TimeSequence[GnssMeasurement],
    z_usbl_tseq: TimeSequence[UsblMeasurement],
) -> tuple[TimeSequence[JointEskfState], TimeSequence[JointEskfState]]:
    meas = _merge_measurements(z_gnss_tseq, z_usbl_tseq)
    return _run_joint_scenario(
        eskf=eskf,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        meas_list=meas,
        desc="Scenario 1 (Joint): USBL",
        include_init_in_upd=False,
        include_init_in_pred=False,
    )


# -----------------------------------------------------------------------------
# Scenario 2: USBL + range
# -----------------------------------------------------------------------------
def run_eskf_s2(
    eskf: ESKF_joint,
    x_init: JointEskfState,
    z_imu_tseq: TimeSequence[ImuMeasurement],
    z_gnss_tseq: TimeSequence[GnssMeasurement],
    z_usbl_tseq: TimeSequence[UsblMeasurement],
    z_range_tseq: TimeSequence[RangeMeasurement],
) -> tuple[TimeSequence[JointEskfState], TimeSequence[JointEskfState]]:
    meas = _merge_measurements(z_gnss_tseq, z_usbl_tseq, z_range_tseq)
    return _run_joint_scenario(
        eskf=eskf,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        meas_list=meas,
        desc="Scenario 2 (Joint): USBL + Range",
        include_init_in_upd=True,
        include_init_in_pred=False,
    )


# -----------------------------------------------------------------------------
# Scenario 3: USBL + range + depth
# -----------------------------------------------------------------------------
def run_eskf_s3(
    eskf: ESKF_joint,
    x_init: JointEskfState,
    z_imu_tseq: TimeSequence[ImuMeasurement],
    z_gnss_tseq: TimeSequence[GnssMeasurement],
    z_usbl_tseq: TimeSequence[UsblMeasurement],
    z_range_tseq: TimeSequence[RangeMeasurement],
    z_depth_tseq: TimeSequence[DepthMeasurement],
) -> tuple[TimeSequence[JointEskfState], TimeSequence[JointEskfState]]:
    meas = _merge_measurements(z_gnss_tseq, z_usbl_tseq, z_range_tseq, z_depth_tseq)
    return _run_joint_scenario(
        eskf=eskf,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        meas_list=meas,
        desc="Scenario 3 (Joint): USBL + Range + Depth",
        include_init_in_upd=True,
        include_init_in_pred=False,
    )