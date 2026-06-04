import numpy as np
from enum import Enum

from quaternion import RotationQuaterion
from senfuslib.timesequence import TimeSequence
from scipy.interpolate import CubicSpline

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
    LINEAR_TURNS - ASV moves linearly between waypoints, with smooth heading
                 changes at each waypoint.  ROV makes a slow linear dive.
                 This is the least favourable scenario for bearing-only, but
                 is simple and smooth, and still has some heading variation.
    """
    CIRCULAR   = "circular"
    FIGURE_8   = "figure_8"
    LINEAR_TURNS = "linear_turns"


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


# ----------------------------------------------------------------------------
# LINEAR + TURNS  (not the best bearing geometry, but simple and smooth)
# ----------------------------------------------------------------------------
# def _generate_linear_turns(t: np.ndarray, g: np.ndarray):
#     """
#     ASV: piecewise linear path with *smooth heading transitions*
#     ROV: smooth linear dive
#     """
#     waypoints = [
#         (np.array([ -60,  -30.0,  0.0]),   0.0),
#         (np.array([-30.0,  -30.0, .0]),  60.0),
#         (np.array([-30.0,  0.0, 0.0]), 120.0),
#         (np.array([-30.0, 40.0, 0.0]), 180.0),
#         (np.array([10.0, 40.0, 0.0]), 240.0),
#         (np.array([10.0,  -10.0,  0.0]), 300.0),
#     ]

#     # -------- Build full ASV trajectory (position + velocity) --------
#     asv_pos = []
#     asv_vel = []

#     for i in range(len(waypoints) - 1):
#         p0, t0 = waypoints[i]
#         p1, t1 = waypoints[i + 1]

#         seg_t = t[(t >= t0) & (t < t1)]
#         vel = (p1 - p0) / (t1 - t0)

#         for ti in seg_t:
#             alpha = (ti - t0) / (t1 - t0)
#             pos = p0 + alpha * (p1 - p0)

#             asv_pos.append(pos)
#             asv_vel.append(vel)

#     asv_pos = np.array(asv_pos)
#     asv_vel = np.array(asv_vel)
#     # Ensure same length as t
#     if len(asv_pos) < len(t):
#         asv_pos = np.vstack([asv_pos, asv_pos[-1]])
#         asv_vel = np.vstack([asv_vel, asv_vel[-1]])

#     # -------- Smooth yaw from velocity --------
#     yaw = np.zeros(len(asv_vel))

#     for i in range(len(asv_vel)):
#         yaw[i] = _ned_yaw_from_velocity(asv_vel[i], yaw[i-1] if i > 0 else 0.0)

#     # unwrap angles to avoid jumps
#     yaw = np.unwrap(yaw)

#     # -------- Build ASV states --------
#     asv_states = [
#         (float(t[i]), AsvNominalState(
#             pos=asv_pos[i],
#             vel=asv_vel[i],
#             ori=RotationQuaterion.from_euler([0.0, 0.0, float(yaw[i])]),
#             accm_bias=np.zeros(3),
#             gyro_bias=np.zeros(3),
#         ))
#         for i in range(len(t))
#     ]

#     # -------- ROV: smooth linear dive --------
#     rov_start = np.array([10.0, -10.0, 8.0])
#     rov_end   = np.array([100.0, -10.0, 18.0])

#     rov_pos = rov_start + (rov_end - rov_start) * (t / t[-1])[:, None]

#     rov_vel = np.zeros_like(rov_pos)
#     dt = t[1] - t[0]
#     rov_vel[1:-1] = (rov_pos[2:] - rov_pos[:-2]) / (2.0 * dt)
#     rov_vel[0] = rov_vel[1]
#     rov_vel[-1] = rov_vel[-2]

#     rov_states = [
#         (float(ti), RovNominalCV(pos=pos, vel=vel))
#         for ti, pos, vel in zip(t, rov_pos, rov_vel)
#     ]

#     print(f"Mean asv velocity: {np.mean(np.linalg.norm(asv_vel, axis=1)):.2f} m/s")
#     print(f"Min-max asv velocity: {np.min(np.linalg.norm(asv_vel, axis=1)):.2f} - {np.max(np.linalg.norm(asv_vel, axis=1)):.2f} m/s")
#     print(f"Mean rov velocity: {np.mean(np.linalg.norm(rov_vel, axis=1)):.2f} m/s")
#     print(f"Min-max rov velocity: {np.min(np.linalg.norm(rov_vel, axis=1)):.2f} - {np.max(np.linalg.norm(rov_vel, axis=1)):.2f} m/s")

#     imu_meas = _imu_from_asv_states(asv_states, t, g)
#     return asv_states, rov_states, imu_meas

# ---------------------------------------------------------------------------
# LINEAR + CIRCULAR-ARC TURNS  (smooth heading, exact target speeds)
# ---------------------------------------------------------------------------

# ── private helpers ──────────────────────────────────────────────────────────

def _circular_arc(p_entry: np.ndarray,
                  dir_in:  np.ndarray,
                  dir_out: np.ndarray,
                  radius:  float,
                  n_pts:   int) -> tuple[np.ndarray, float]:
    cross_z = dir_in[0] * dir_out[1] - dir_in[1] * dir_out[0]
    sign    = np.sign(cross_z)          # +1 = left / CCW,  -1 = right / CW

    dangle = (np.arctan2(dir_out[1], dir_out[0])
              - np.arctan2(dir_in[1],  dir_in[0]))
    dangle = (dangle + np.pi) % (2 * np.pi) - np.pi   # wrap to (-π, π]

    # Centre of curvature is perpendicular to dir_in, on the inside of the turn
    perp   = np.array([-dir_in[1], dir_in[0], 0.0])
    center = p_entry + perp * sign * radius

    dp     = p_entry - center
    angles = np.linspace(0.0, dangle, n_pts)
    pts    = np.empty((n_pts, 3))
    for i, a in enumerate(angles):
        ca, sa = np.cos(a), np.sin(a)
        pts[i] = center + np.array([
            ca * dp[0] - sa * dp[1],
            sa * dp[0] + ca * dp[1],
            0.0,
        ])

    return pts, abs(dangle) * radius


def _sample_straight(p_start: np.ndarray,
                     direction: np.ndarray,
                     length: float,
                     spacing: float) -> np.ndarray:
    """Uniformly-spaced points along a straight segment (open end)."""
    n = max(2, int(length / spacing))
    ts = np.linspace(0.0, 1.0, n, endpoint=False)
    return p_start + np.outer(ts, direction * length)


# ── public function ──────────────────────────────────────────────────────────

def _generate_linear_turns(t: np.ndarray, g: np.ndarray):
    """
    ASV: piecewise path with *smooth circular-arc turns*.
    ROV: smooth straight-line dive.

    Parameters
    ----------
    t : time array [s], shape (N,), uniform spacing dt = t[1] - t[0]
    g : gravity vector in NED  [m/s²], e.g. np.array([0, 0, 9.82])

    Returns
    -------
    asv_states : list of (t_i, AsvNominalState)
    rov_states : list of (t_i, RovNominalCV)
    imu_meas   : list of (t_i, ImuMeasurement)  — from _imu_from_asv_states
    """
    ASV_SPEED = 1.5   # m/s  — target mean (and instantaneous) speed
    ROV_SPEED = 0.5   # m/s
    TURN_R    = 20.0  # circular-arc radius [m]
    PT_SPACING = 1.5e-3  # 1.5 mm between table points — fine enough for dt=0.01 s

    dt = float(t[1] - t[0])

    # ------------------------------------------------------------------
    # 1.  Build the ASV path geometry
    #     North → (arc R=20m) → East → (arc R=20m) → North
    # ------------------------------------------------------------------
    arc_len = TURN_R * (np.pi / 2)               # ≈ 31.42 m per 90° arc
    seg_len = (ASV_SPEED * t[-1] - 2 * arc_len) / 3  # ≈ 129.1 m per straight leg

    dir_N = np.array([1.0, 0.0, 0.0])
    dir_E = np.array([0.0, 1.0, 0.0])

    # Key positions
    p0 = np.zeros(3)                    # start
    p1 = p0 + dir_N * seg_len           # arc 1 entry
    n_arc = max(2, int(arc_len / PT_SPACING))
    arc1_pts, _ = _circular_arc(p1, dir_N, dir_E, TURN_R, n_arc)
    p2 = arc1_pts[-1]                   # arc 1 exit / leg 2 start

    p3 = p2 + dir_E * seg_len           # arc 2 entry
    arc2_pts, _ = _circular_arc(p3, dir_E, dir_N, TURN_R, n_arc)
    p4 = arc2_pts[-1]                   # arc 2 exit / leg 3 start

    p5 = p4 + dir_N * seg_len           # end

    # ------------------------------------------------------------------
    # 2.  Concatenate into one dense position table
    # ------------------------------------------------------------------
    pieces = [
        _sample_straight(p0, dir_N, seg_len, PT_SPACING),
        arc1_pts,
        _sample_straight(p2, dir_E, seg_len, PT_SPACING),
        arc2_pts,
        _sample_straight(p4, dir_N, seg_len, PT_SPACING),
        p5[None, :],   # close the path
    ]
    all_pts = np.vstack(pieces)

    ds    = np.linalg.norm(np.diff(all_pts, axis=0), axis=1)
    arc_s = np.concatenate([[0.0], np.cumsum(ds)])
    total_arc = arc_s[-1]

    # ------------------------------------------------------------------
    # 3.  Arc-length reparameterise at constant ASV_SPEED
    # ------------------------------------------------------------------
    asv_pos = np.empty((len(t), 3))
    for i, tv in enumerate(t):
        s   = min(ASV_SPEED * tv, total_arc - 1e-9)
        idx = int(np.searchsorted(arc_s, s))
        asv_pos[i] = all_pts[max(1, min(idx, len(arc_s) - 1))]

    # Velocity via central finite differences, then re-normalise to exact speed
    asv_vel = np.empty_like(asv_pos)
    asv_vel[1:-1] = (asv_pos[2:] - asv_pos[:-2]) / (2.0 * dt)
    asv_vel[0]    = asv_vel[1]
    asv_vel[-1]   = asv_vel[-2]
    spd = np.linalg.norm(asv_vel, axis=1, keepdims=True)
    asv_vel = asv_vel / np.where(spd > 1e-9, spd, 1.0) * ASV_SPEED

    # ------------------------------------------------------------------
    # 4.  ROV: straight-line constant-velocity dive
    #     Direction: mostly North-East with a gentle descent
    # ------------------------------------------------------------------
    rov_start = np.array([ 200.0, 100.0,  8.0])
    raw_dir   = np.array([  0.0,   -1.0,  0.1])
    rov_vel_vec = raw_dir / np.linalg.norm(raw_dir) * ROV_SPEED
    rov_end   = rov_start + rov_vel_vec * t[-1]

    rov_pos = rov_start + rov_vel_vec * t[:, None]
    rov_vel = np.tile(rov_vel_vec, (len(t), 1))

    # ------------------------------------------------------------------
    # 5.  Compute smooth yaw from velocity
    # ------------------------------------------------------------------
    yaw = np.zeros(len(asv_vel))
    for i in range(len(asv_vel)):
        yaw[i] = _ned_yaw_from_velocity(
            asv_vel[i],
            yaw[i - 1] if i > 0 else 0.0
        )

    yaw_smooth = np.unwrap(yaw)

    # ------------------------------------------------------------------
    # 6.  Build state sequences
    # ------------------------------------------------------------------


    asv_states = [
        (float(t[i]), AsvNominalState(
            pos=asv_pos[i],
            vel=asv_vel[i],
            ori=RotationQuaterion.from_euler([0.0, 0.0, float(yaw_smooth[i])]),
            accm_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
        ))
        for i in range(len(t))
    ]

    rov_states = [
        (float(t[i]), RovNominalCV(pos=rov_pos[i], vel=rov_vel[i]))
        for i in range(len(t))
    ]

    imu_meas = _imu_from_asv_states(asv_states, t, g)

    return asv_states, rov_states, imu_meas

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
# Public entry-point
# ---------------------------------------------------------------------------

_GENERATORS = {
    TrajectoryType.CIRCULAR:   _generate_circular,
    TrajectoryType.FIGURE_8:   _generate_figure_8,
    TrajectoryType.LINEAR_TURNS: _generate_linear_turns,
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
