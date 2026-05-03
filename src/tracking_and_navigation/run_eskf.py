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


def _merge_other_measurements(*tseqs: TimeSequence):
    """
    Merge non-GNSS measurement streams into a sorted list [(t, z), ...].
    """
    all_meas = []
    for ts in tseqs:
        if ts is None:
            continue
        all_meas += list(ts.items())
    all_meas.sort(key=lambda x: x[0])
    return all_meas


def _run_preint_scenario(
    eskf: ESKF_joint,
    x_init: JointEskfState,
    z_imu_tseq: TimeSequence[ImuMeasurement],
    z_gnss_tseq: TimeSequence[GnssMeasurement],
    other_meas_list,  # sorted [(t, z)] of non-GNSS measurements
    desc: str,
) -> tuple[TimeSequence[JointEskfState], TimeSequence[JointEskfState]]:
    """
    GNSS-driven loop with IMU preintegration.

    The filter predicts and updates only at GNSS measurement times.
    Non-GNSS measurements (USBL, range, depth) that arrive between two GNSS
    events are buffered causally and applied in time order at the next GNSS
    event, before the GNSS update itself.

    Between GNSS events all IMU samples are collected and passed to
    eskf.preintegrate_imu(), which composes F and Q along the nominal
    trajectory and applies a single 21×21 covariance update.
    """
    imu_list = list(z_imu_tseq.items())
    imu_len = len(imu_list)
    if not imu_list:
        raise ValueError("z_imu_tseq is empty")

    t_prev = imu_list[0][0]
    x_prev = x_init
    imu_idx = 0
    other_idx = 0
    other_len = len(other_meas_list)

    upd_tseq = TimeSequence()
    pred_tseq = TimeSequence([(t_prev, x_init)])

    for t_gnss, z_gnss in tqdm(z_gnss_tseq.items(), desc=desc):
        if t_gnss < t_prev:
            continue

        # 1) Collect all IMU samples up to this GNSS time for preintegration
        imu_steps: list[tuple[ImuMeasurement, float]] = []
        while imu_idx < imu_len and imu_list[imu_idx][0] <= t_gnss:
            t_imu, z_imu = imu_list[imu_idx]
            dt = t_imu - t_prev
            if dt > 0:
                imu_steps.append((z_imu, dt))
                t_prev = t_imu
            imu_idx += 1

        # Bridge any gap between last IMU and GNSS time
        if t_gnss > t_prev:
            last_z_imu = imu_list[max(0, imu_idx - 1)][1]
            imu_steps.append((last_z_imu, t_gnss - t_prev))
            t_prev = t_gnss

        # 2) Single covariance propagation over all accumulated IMU steps
        x_pred = eskf.preintegrate_imu(x_prev, imu_steps)
        if t_gnss not in pred_tseq:
            pred_tseq.insert(t_gnss, x_pred)

        # 3) Buffer non-GNSS measurements that arrived before this GNSS event
        #    and apply them in causal (time) order before the GNSS update
        buffered: list[tuple[float, object]] = []
        while other_idx < other_len and other_meas_list[other_idx][0] <= t_gnss:
            buffered.append(other_meas_list[other_idx])
            other_idx += 1

        x_cur = x_pred
        for _t, z_m in buffered:
            if isinstance(z_m, UsblMeasurement):
                x_cur, _ = eskf.update_from_usbl(x_cur, z_m)
            elif isinstance(z_m, RangeMeasurement):
                x_cur, _ = eskf.update_from_range(x_cur, z_m)
            elif isinstance(z_m, DepthMeasurement):
                x_cur, _ = eskf.update_from_depth(x_cur, z_m)

        # 4) GNSS update last
        x_upd, _ = eskf.update_from_gnss_asv(x_cur, z_gnss)
        if t_gnss not in upd_tseq:
            upd_tseq.insert(t_gnss, x_upd)
        x_prev = x_upd

    return upd_tseq, pred_tseq


# -----------------------------------------------------------------------------
# Scenario 1: GNSS + USBL
# -----------------------------------------------------------------------------
def run_eskf_s1(
    eskf: ESKF_joint,
    x_init: JointEskfState,
    z_imu_tseq: TimeSequence[ImuMeasurement],
    z_gnss_tseq: TimeSequence[GnssMeasurement],
    z_usbl_tseq: TimeSequence[UsblMeasurement],
) -> tuple[TimeSequence[JointEskfState], TimeSequence[JointEskfState]]:
    other = _merge_other_measurements(z_usbl_tseq)
    return _run_preint_scenario(
        eskf=eskf,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        other_meas_list=other,
        desc="Scenario 1: GNSS + USBL (preint)",
    )


# -----------------------------------------------------------------------------
# Scenario 2: GNSS + USBL + range
# -----------------------------------------------------------------------------
def run_eskf_s2(
    eskf: ESKF_joint,
    x_init: JointEskfState,
    z_imu_tseq: TimeSequence[ImuMeasurement],
    z_gnss_tseq: TimeSequence[GnssMeasurement],
    z_usbl_tseq: TimeSequence[UsblMeasurement],
    z_range_tseq: TimeSequence[RangeMeasurement],
) -> tuple[TimeSequence[JointEskfState], TimeSequence[JointEskfState]]:
    other = _merge_other_measurements(z_usbl_tseq, z_range_tseq)
    return _run_preint_scenario(
        eskf=eskf,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        other_meas_list=other,
        desc="Scenario 2: GNSS + USBL + Range (preint)",
    )


# -----------------------------------------------------------------------------
# Scenario 3: GNSS + USBL + range + depth
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
    other = _merge_other_measurements(z_usbl_tseq, z_range_tseq, z_depth_tseq)
    return _run_preint_scenario(
        eskf=eskf,
        x_init=x_init,
        z_imu_tseq=z_imu_tseq,
        z_gnss_tseq=z_gnss_tseq,
        other_meas_list=other,
        desc="Scenario 3: GNSS + USBL + Range + Depth (preint)",
    )