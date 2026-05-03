import numpy as np
from enum import Enum

from quaternion import RotationQuaterion
from senfuslib.timesequence import TimeSequence

from tracking_and_navigation.states import AsvNominalState, RovNominalCV
from tracking_and_navigation.measurements import ImuMeasurement


class TrajectoryType(str, Enum):
    """
    Trajectory mode selector for both USV and ROV.

    CIRCULAR   - ASV circles a fixed centre; ROV follows piecewise-linear
                 waypoints.  The CV model matches ROV truth exactly, making
                 this the most favourable scenario for the ESKF.

    FIGURE_8   - ASV traces a Lissajous figure-8 (2:1 frequency ratio),
                 producing frequent heading reversals that improve bearing-
                 only observability.  ROV makes a slow circular sweep with
                 sinusoidal depth — smooth but non-CV, introducing mild model
                 mismatch.

    SINUSOIDAL - ASV follows a sinusoidal S-curve (maximum heading variation,
                 best bearing geometry).  ROV makes a maneuvering 3-D path
                 with abrupt speed changes (strong CV violation), exposing
                 the scenario where FGO's selective re-linearisation and
                 delayed-measurement handling give the most benefit.
    """
    CIRCULAR   = "circular"
    FIGURE_8   = "figure_8"
    SINUSOIDAL = "sinusoidal"


def _ned_yaw_from_velocity(v_ned: np.ndarray, fallback_yaw: float = 0.0) -> float:
    vn, ve = float(v_ned[0]), float(v_ned[1])
    if vn * vn + ve * ve < 1e-10:
        return fallback_yaw
    return float(np.arctan2(ve, vn))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rov_from_waypoints(t: np.ndarray, waypoints: list) -> list:
    """Piecewise constant-velocity ROV from a list of (pos, time) waypoints."""
    rov_states = []
    for i in range(len(waypoints) - 1):
        p0, t0 = waypoints[i]
        p1, t1 = waypoints[i + 1]
        seg_vel = (p1 - p0) / (t1 - t0)
        seg_t = t[(t >= t0) & (t < t1)]
        for ti in seg_t:
            alpha = (ti - t0) / (t1 - t0)
            pos = p0 + alpha * (p1 - p0)
            rov_states.append((float(ti), RovNominalCV(pos=pos, vel=seg_vel)))
    # Append terminal state if needed
    if not rov_states or rov_states[-1][0] < t[-1] - 1e-9:
        p_last, t_last = waypoints[-1]
        p_prev, t_prev = waypoints[-2]
        v_last = (p_last - p_prev) / (t_last - t_prev)
        rov_states.append((float(t[-1]), RovNominalCV(pos=p_last, vel=v_last)))
    return rov_states


def _imu_from_asv_states(asv_states: list, t: np.ndarray, g: np.ndarray) -> list:
    """
    Derive ideal IMU measurements from a list of (ti, AsvNominalState).
    Central-difference velocity derivative gives world-frame acceleration;
    body-frame specific force and body angular rate are computed from that.
    """
    n = len(asv_states)
    dt = float(t[1] - t[0]) if n > 1 else 0.1
    imu_meas = []
    for i in range(n):
        ti, s = asv_states[i]
        R_nb = s.ori.as_rotmat()
        R_bn = R_nb.T

        if i == 0 or i == n - 1:
            acc_w = np.zeros(3)
        else:
            v_prev = np.asarray(asv_states[i - 1][1].vel)
            v_next = np.asarray(asv_states[i + 1][1].vel)
            acc_w = (v_next - v_prev) / (2.0 * dt)

        f_b = R_bn @ (acc_w - g)

        if i == 0 or i == n - 1:
            omega_b = np.zeros(3)
        else:
            R_prev = asv_states[i - 1][1].ori.as_rotmat()
            R_next = asv_states[i + 1][1].ori.as_rotmat()
            yaw_prev = float(np.arctan2(R_prev[1, 0], R_prev[0, 0]))
            yaw_next = float(np.arctan2(R_next[1, 0], R_next[0, 0]))
            dyaw = (yaw_next - yaw_prev + np.pi) % (2.0 * np.pi) - np.pi
            omega_b = np.array([0.0, 0.0, dyaw / (2.0 * dt)])

        imu_meas.append((float(ti), ImuMeasurement(acc=f_b, avel=omega_b)))
    return imu_meas


# ---------------------------------------------------------------------------
# CIRCULAR  (original — most ideal for ESKF)
# ---------------------------------------------------------------------------

def _generate_circular(t: np.ndarray, g: np.ndarray):
    """
    ASV: circular orbit around a fixed centre.
    ROV: piecewise constant-velocity waypoints (exact CV model match).
    """
    center = np.array([30.0, 0.0, 0.0])
    radius = 25.0
    omega  = 0.05  # rad/s → period ≈ 126 s

    asv_pos = np.stack([
        center[0] + radius * np.cos(omega * t),
        center[1] + radius * np.sin(omega * t),
        np.zeros_like(t),
    ], axis=1)
    asv_vel = np.stack([
        -radius * omega * np.sin(omega * t),
        +radius * omega * np.cos(omega * t),
        np.zeros_like(t),
    ], axis=1)
    asv_yaw = np.array([_ned_yaw_from_velocity(v) for v in asv_vel])

    asv_states = [
        (float(t[i]), AsvNominalState(
            pos=asv_pos[i], vel=asv_vel[i],
            ori=RotationQuaterion.from_euler([0.0, 0.0, float(asv_yaw[i])]),
            accm_bias=np.zeros(3), gyro_bias=np.zeros(3),
        ))
        for i in range(len(t))
    ]

    waypoints = [
        (np.array([ 0.0,  0.0,  5.0]),   0.0),
        (np.array([20.0,  5.0, 10.0]),  60.0),
        (np.array([40.0,  0.0, 15.0]), 120.0),
        (np.array([40.0, 20.0, 20.0]), 180.0),
        (np.array([20.0, 20.0, 12.0]), 240.0),
        (np.array([ 0.0,  0.0,  5.0]), 300.0),
    ]
    rov_states = _rov_from_waypoints(t, waypoints)
    imu_meas   = _imu_from_asv_states(asv_states, t, g)
    return asv_states, rov_states, imu_meas


# ---------------------------------------------------------------------------
# FIGURE-8  (better bearing-only geometry; mild CV violation for ROV)
# ---------------------------------------------------------------------------

def _generate_figure_8(t: np.ndarray, g: np.ndarray):
    """
    ASV: Lissajous figure-8 (N: sin(wt), E: sin(2wt)) — frequent heading
    reversals give strong bearing-only excitation without singularities.

    ROV: slow horizontal circle with sinusoidal depth.  The smooth but
    non-piecewise-linear motion introduces mild model mismatch vs CV.
    """
    center  = np.array([30.0, 0.0, 0.0])
    amp_n   = 30.0    # N amplitude [m]
    amp_e   = 20.0    # E amplitude [m]
    omega   = 0.05    # rad/s → 1 full figure-8 every ≈126 s

    asv_n = center[0] + amp_n * np.sin(omega * t)
    asv_e = center[1] + amp_e * np.sin(2.0 * omega * t)

    asv_pos = np.stack([asv_n, asv_e, np.zeros_like(t)], axis=1)
    asv_vel = np.stack([
        amp_n * omega * np.cos(omega * t),
        2.0 * amp_e * omega * np.cos(2.0 * omega * t),
        np.zeros_like(t),
    ], axis=1)
    asv_yaw = np.array([_ned_yaw_from_velocity(v) for v in asv_vel])

    asv_states = [
        (float(t[i]), AsvNominalState(
            pos=asv_pos[i], vel=asv_vel[i],
            ori=RotationQuaterion.from_euler([0.0, 0.0, float(asv_yaw[i])]),
            accm_bias=np.zeros(3), gyro_bias=np.zeros(3),
        ))
        for i in range(len(t))
    ]

    # ROV: slow horizontal circular arc + sinusoidal depth
    rov_r     = 15.0
    rov_omega = 0.015   # rad/s
    rov_cx, rov_cy = 20.0, 10.0
    depth_mean, depth_amp = 12.0, 6.0

    rov_n = rov_cx + rov_r * np.cos(rov_omega * t)
    rov_e = rov_cy + rov_r * np.sin(rov_omega * t)
    rov_d = depth_mean + depth_amp * np.sin(0.012 * t)

    rov_pos = np.stack([rov_n, rov_e, rov_d], axis=1)
    rov_vel = np.zeros_like(rov_pos)
    rov_vel[1:-1] = (rov_pos[2:] - rov_pos[:-2]) / (2.0 * (t[1] - t[0]))
    rov_vel[0]  = rov_vel[1]
    rov_vel[-1] = rov_vel[-2]

    rov_states = [
        (float(t[i]), RovNominalCV(pos=rov_pos[i], vel=rov_vel[i]))
        for i in range(len(t))
    ]
    imu_meas = _imu_from_asv_states(asv_states, t, g)
    return asv_states, rov_states, imu_meas


# ---------------------------------------------------------------------------
# SINUSOIDAL  (hardest for ESKF; best for FGO)
# ---------------------------------------------------------------------------

def _generate_sinusoidal(t: np.ndarray, g: np.ndarray):
    """
    ASV: straight-line forward with sinusoidal cross-track (S-curve / slalom).
    Maximum heading rate variation → strongest bearing-only geometry change.

    ROV: piecewise path with abrupt speed changes between waypoints.  The
    constant-velocity model is violated at every segment transition, which
    is where the FGO's selective re-linearisation gives the most benefit.
    The varying inter-waypoint speeds also stress acoustic delay handling.
    """
    v_fwd  = 1.5    # m/s forward speed
    amp_e  = 20.0   # m cross-track amplitude
    freq_e = 0.025  # rad/s (period ≈ 251 s)

    asv_n = v_fwd * t
    asv_e = amp_e * np.sin(freq_e * t)

    asv_pos = np.stack([asv_n, asv_e, np.zeros_like(t)], axis=1)
    asv_vel = np.stack([
        np.full_like(t, v_fwd),
        amp_e * freq_e * np.cos(freq_e * t),
        np.zeros_like(t),
    ], axis=1)
    asv_yaw = np.array([_ned_yaw_from_velocity(v) for v in asv_vel])

    asv_states = [
        (float(t[i]), AsvNominalState(
            pos=asv_pos[i], vel=asv_vel[i],
            ori=RotationQuaterion.from_euler([0.0, 0.0, float(asv_yaw[i])]),
            accm_bias=np.zeros(3), gyro_bias=np.zeros(3),
        ))
        for i in range(len(t))
    ]

    # ROV: waypoints with deliberately varied speeds (strong CV violation at junctions)
    waypoints = [
        (np.array([ 5.0, -5.0, 10.0]),   0.0),
        (np.array([25.0,  0.0, 15.0]),  50.0),   # fast sprint
        (np.array([27.0,  5.0, 18.0]),  80.0),   # slow crawl
        (np.array([40.0, 15.0, 22.0]), 130.0),   # medium speed
        (np.array([50.0, 10.0, 12.0]), 180.0),   # fast descent
        (np.array([60.0,  5.0,  8.0]), 220.0),   # medium, surfacing
        (np.array([75.0, 20.0, 20.0]), 300.0),   # slow, dive again
    ]
    rov_states = _rov_from_waypoints(t, waypoints)
    imu_meas   = _imu_from_asv_states(asv_states, t, g)
    return asv_states, rov_states, imu_meas


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

_GENERATORS = {
    TrajectoryType.CIRCULAR:   _generate_circular,
    TrajectoryType.FIGURE_8:   _generate_figure_8,
    TrajectoryType.SINUSOIDAL: _generate_sinusoidal,
}


def generate_trajectories(
    duration: float = 300.0,
    dt: float = 0.1,
    trajectory_type: TrajectoryType = TrajectoryType.CIRCULAR,
):
    """
    Generate ground-truth NED trajectories for the ASV and ROV, plus ideal
    IMU measurements for the ASV.

    Args:
        duration:         Simulation length [s]
        dt:               Time step [s]
        trajectory_type:  One of TrajectoryType.{CIRCULAR, FIGURE_8, SINUSOIDAL}

    Returns:
        asv_gt_tseq : TimeSequence[AsvNominalState]
        rov_gt_tseq : TimeSequence[RovNominalCV]
        imu_tseq    : TimeSequence[ImuMeasurement]
    """
    t = np.arange(0.0, duration + 1e-12, dt)
    g = np.array([0.0, 0.0, 9.82])  # NED gravity (down positive)

    gen_fn = _GENERATORS[TrajectoryType(trajectory_type)]
    asv_states, rov_states, imu_meas = gen_fn(t, g)

    return (
        TimeSequence(asv_states),
        TimeSequence(rov_states),
        TimeSequence(imu_meas),
    )
