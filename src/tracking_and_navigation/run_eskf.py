import numpy as np
from tqdm import tqdm

from quaternion import RotationQuaterion
from senfuslib import TimeSequence

from tracking_and_navigation.states import AsvNominalState, JointEskfState, JointNominalState, RovNominalCV
from tracking_and_navigation.eskf import ESKF_joint

from tracking_and_navigation.measurements import (
    GnssMeasurement,
    ImuMeasurement,
    UsblMeasurement,
    RangeMeasurement,
    DepthMeasurement,
)
from utils.angles import wrap_to_2pi, wrap_to_pi
from utils.withXYZ import WithXYZ


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
) -> tuple[
    TimeSequence[JointEskfState], 
    TimeSequence[JointEskfState], 
    dict[str, tuple[TimeSequence, TimeSequence]]
]:
    """
    Run joint filter driven by IMU, with async updates (USBL/range/depth).

    Return (upd_tseq, pred_tseq, z_preds) where z_preds is a dict mapping
    sensor name to (z_pred_tseq, z_meas_tseq) for NIS computation.

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

    # Buffers for NIS computation
    _gnss_buf = []
    _usbl_buf = []
    _range_buf = []
    _depth_buf = []

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
                x_upd, z_pred = eskf.update_from_gnss_asv(x_pred, z_m)
                _gnss_buf.append((t_m, z_pred, z_m))
            elif isinstance(z_m, UsblMeasurement):
                x_upd, z_pred = eskf.update_from_usbl(x_pred, z_m)
                _usbl_buf.append((t_m, z_pred, z_m))
            elif isinstance(z_m, RangeMeasurement):
                x_upd, z_pred = eskf.update_from_range(x_pred, z_m)
                _range_buf.append((t_m, z_pred, z_m))
            elif isinstance(z_m, DepthMeasurement):
                x_upd, z_pred = eskf.update_from_depth(x_pred, z_m)
                _depth_buf.append((t_m, z_pred, z_m))
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

    def _to_tseqs(lst):
        zp = TimeSequence()
        zm = TimeSequence()
        for t, z_pred, z_meas in lst:
            if t not in zp:
                zp.insert(t, z_pred)
                zm.insert(t, z_meas)
        return zp, zm

    z_preds = {k: _to_tseqs(v) for k, v in [
        ('gnss',  _gnss_buf),
        ('usbl',  _usbl_buf),
        ('range', _range_buf),
        ('depth', _depth_buf),
    ] if v}

    return upd_tseq, pred_tseq, z_preds


# -----------------------------------------------------------------------------
# Scenario 1: USBL bearing only
# -----------------------------------------------------------------------------
def run_eskf_s1(
    eskf: ESKF_joint,
    x_init: JointEskfState,
    z_imu_tseq: TimeSequence[ImuMeasurement],
    z_gnss_tseq: TimeSequence[GnssMeasurement],
    z_usbl_tseq: TimeSequence[UsblMeasurement],
) -> tuple[
    TimeSequence[JointEskfState],
    TimeSequence[JointEskfState],
    dict[str, tuple[TimeSequence, TimeSequence]],
]:
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
) -> tuple[
    TimeSequence[JointEskfState],
    TimeSequence[JointEskfState],
    dict[str, tuple[TimeSequence, TimeSequence]],
]:
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
) -> tuple[
    TimeSequence[JointEskfState],
    TimeSequence[JointEskfState],
    dict[str, tuple[TimeSequence, TimeSequence]],
]:
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

def init_asv_from_gnss(
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

def init_rov_from_usbl_range_depth(
    x_init: JointEskfState, # Initial guess (can be from ASV init or just default)
    x_asv: AsvNominalState, # ASV state (after init from GNSS or in case we init runtime)
    z_usbl_tseq,
    z_range_tseq,
    z_depth_tseq,
    range_guess=10.0,
    usbl_lever_arm=np.array([0.0, 0.0, 1.2]), # Assuming USBL is at ASV center for simplicity
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
            depth,  # depth is positive down, but we want z positive up
        ])

    rov_init = RovNominalCV(
        pos=rov_pos,
        vel=rov_vel,
    )
    return JointEskfState(
        nom=JointNominalState(asv=x_asv, rov=rov_init),
        err=x_init.err,
    )