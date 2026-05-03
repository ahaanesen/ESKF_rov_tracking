import numpy as np
from scipy.interpolate import splprep, splev

from quaternion import RotationQuaterion
from senfuslib.timesequence import TimeSequence

from tracking_and_navigation.states import AsvNominalState, RovNominalCV
from tracking_and_navigation.measurements import ImuMeasurement


def _ned_yaw_from_velocity(v_ned: np.ndarray, fallback_yaw: float = 0.0) -> float:
    """
    Compute yaw from NED horizontal velocity.
    If speed is near zero, return fallback_yaw.
    """
    vn, ve = float(v_ned[0]), float(v_ned[1])
    if vn * vn + ve * ve < 1e-10:
        return fallback_yaw
    return float(np.arctan2(ve, vn))


def generate_trajectories(duration: float = 300.0, dt: float = 0.1):
    """
    Generate ground-truth trajectories in NED for:
      - ASV: ASVNominalState (pos, vel, ori, biases=0)
      - ROV: ROVNominalCV    (pos, vel)

    Also generates an ideal IMU sequence for the ASV (body-frame specific force and body rates),
    consistent with the generated ASV motion.

    Returns:
        asv_gt_tseq: TimeSequence[ASVNominalState]
        rov_gt_tseq: TimeSequence[ROVNominalCV]
        imu_tseq:    TimeSequence[ImuMeasurement]
    """
    t = np.arange(0.0, duration + 1e-12, dt)

    # -------------------------------------------------------------------------
    # ASV trajectory: orbit around an offset center (never directly above ROV)
    # Speed oscillates to improve bearing observability.
    # -------------------------------------------------------------------------
    asv_center = np.array([30.0, 0.0, 0.0])
    asv_radius = 25.0
    asv_omega_base = 0.05   # mean angular velocity, rad/s
    asv_omega_amp  = 0.02   # oscillation amplitude, rad/s  (~40 % of base)
    asv_omega_mod  = 0.10   # oscillation frequency, rad/s  (~63 s period)

    # Instantaneous angular velocity: omega(t) = omega_base + omega_amp*sin(omega_mod*t)
    asv_omega_t = asv_omega_base + asv_omega_amp * np.sin(asv_omega_mod * t)

    # Angular acceleration: d(omega)/dt = omega_amp * omega_mod * cos(omega_mod*t)
    asv_alpha_t = asv_omega_amp * asv_omega_mod * np.cos(asv_omega_mod * t)

    # Heading angle: integral of omega(t)
    # theta(t) = omega_base*t - (omega_amp/omega_mod)*cos(omega_mod*t) + C
    # Choose C so theta(0)=0: C = omega_amp/omega_mod
    asv_theta = (
        asv_omega_base * t
        - (asv_omega_amp / asv_omega_mod) * np.cos(asv_omega_mod * t)
        + (asv_omega_amp / asv_omega_mod)
    )

    # Position in NED
    asv_pos = np.stack(
        [
            asv_center[0] + asv_radius * np.cos(asv_theta),
            asv_center[1] + asv_radius * np.sin(asv_theta),
            np.zeros_like(t),
        ],
        axis=1,
    )

    # Velocity in NED: d/dt [R cos(theta), R sin(theta)] = R * omega(t) * [-sin, cos]
    asv_vel = np.stack(
        [
            -asv_radius * asv_omega_t * np.sin(asv_theta),
            +asv_radius * asv_omega_t * np.cos(asv_theta),
            np.zeros_like(t),
        ],
        axis=1,
    )

    # Acceleration in NED: centripetal + tangential
    asv_acc = np.stack(
        [
            -asv_radius * asv_omega_t**2 * np.cos(asv_theta)
            - asv_radius * asv_alpha_t  * np.sin(asv_theta),
            -asv_radius * asv_omega_t**2 * np.sin(asv_theta)
            + asv_radius * asv_alpha_t  * np.cos(asv_theta),
            np.zeros_like(t),
        ],
        axis=1,
    )

    # Yaw aligned with velocity direction (tangent to circle): theta + pi/2
    asv_yaw = asv_theta + np.pi / 2

    # Build ASV states (biases = 0 ground truth)
    asv_states = []
    for i, ti in enumerate(t):
        asv_states.append(
            (
                float(ti),
                AsvNominalState(
                    pos=asv_pos[i],
                    vel=asv_vel[i],
                    ori=RotationQuaterion.from_euler([0.0, 0.0, float(asv_yaw[i])]),
                    accm_bias=np.zeros(3),
                    gyro_bias=np.zeros(3),
                ),
            )
        )

    # -------------------------------------------------------------------------
    # ASV ideal IMU: body angular rates + specific force in body frame
    # -------------------------------------------------------------------------
    # Convention consistent with your ModelIMU.predict_nom:
    #   acc_world = R(q) @ acc_body + g
    # => acc_body = R(q)^T @ (acc_world - g)
    #
    # Here, acc_world is asv_acc (NED), g is [0,0,9.82] (down positive in NED).
    g = np.array([0.0, 0.0, 9.82])

    imu_meas = []
    for i, ti in enumerate(t):
        q = asv_states[i][1].ori
        R_nb = q.as_rotmat()          # likely nav-from-body, based on your ModelIMU usage
        R_bn = R_nb.T

        # specific force in body
        f_b = R_bn @ (asv_acc[i] - g)

        # body angular rate:
        # since roll=pitch=0 and yaw changes smoothly, omega_b approx [0,0,yaw_rate]
        # yaw_rate is constant here = omega (for perfect circle), in NED.
        # For small roll/pitch, omega_body ≈ [0,0, yaw_rate]
        omega_b = np.array([0.0, 0.0, asv_omega_t[i]])

        imu_meas.append((float(ti), ImuMeasurement(acc=f_b, avel=omega_b)))

    # -------------------------------------------------------------------------
    # ROV ground truth: smooth spline path with constant speed
    # -------------------------------------------------------------------------

    waypoints_new = [
        np.array([0.0, 0.0, 5.0]),
        np.array([10.0, 20.0, 10.0]),
        np.array([20.0, 10.0, 15.0]),
        np.array([30.0, 20.0, 20.0]),
        np.array([40.0, 0.0, 25.0]),
        np.array([30.0, -20.0, 18.0]),
        np.array([20.0, -10.0, 15.0]),
        np.array([10.0, -20.0, 12.0]),
        np.array([0.0, 0.0, 5.0]),
    ]

    pts = np.array(waypoints_new)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Build spline

    tck, u = splprep([x, y, z], s=0, k=3)

    # High‑res sampling for arc length
    u_fine = np.linspace(0, 1, 5000)
    x_f, y_f, z_f = splev(u_fine, tck)

    diffs = np.sqrt(np.diff(x_f)**2 + np.diff(y_f)**2 + np.diff(z_f)**2)
    arc_length = np.concatenate(([0], np.cumsum(diffs)))
    arc_length_norm = arc_length / arc_length[-1]

    # Constant‑speed parameterization for the simulation times
    t_norm = np.zeros_like(t)
    if t[-1] > t[0]:
           t_norm = (t - t[0]) / (t[-1] - t[0])
    u_const = np.interp(t_norm, arc_length_norm, u_fine)

    # Evaluate spline at constant speed
    x_c, y_c, z_c = splev(u_const, tck)

    # Compute velocity by differentiating spline
    dx_du, dy_du, dz_du = splev(u_const, tck, der=1)
    du_dt = np.gradient(u_const, t, edge_order=2)
    dx = dx_du * du_dt
    dy = dy_du * du_dt
    dz = dz_du * du_dt

    rov_states = []
    for i, ti in enumerate(t):
        pos = np.array([x_c[i], y_c[i], z_c[i]])
        vel = np.array([dx[i], dy[i], dz[i]])

        rov_states.append(
            (
                float(ti),
                RovNominalCV(
                    pos=pos,
                    vel=vel,
                ),
            )
        )

    return TimeSequence(asv_states), TimeSequence(rov_states), TimeSequence(imu_meas)

def asv_circle_traj(t: np.ndarray, duration: float = 300.0, dt: float = 0.1):
    # ASV trajectory to maximize bearing observability: circle around ROV, with constant speed
    asv_center = np.array([30.0, 0.0, 0.0])
    asv_radius = 25.0
    asv_omega = 0.05  # rad/s

    # Position in NED
    # N = centerN + R cos(wt)
    # E = centerE + R sin(wt)
    # D = 0
    asv_pos = np.stack(
        [ 
            asv_center[0] + asv_radius * np.cos(asv_omega * t),
            asv_center[1] + asv_radius * np.sin(asv_omega * t),
            np.zeros_like(t),
        ],
        axis=1,
    )

    # Velocity in NED: derivative of position
    asv_vel = np.stack(
        [
            -asv_radius * asv_omega * np.sin(asv_omega * t),
            +asv_radius * asv_omega * np.cos(asv_omega * t),
            np.zeros_like(t),
        ],
        axis=1,
    )

    # Acceleration in NED: derivative of velocity
    asv_acc = np.stack(
        [
            -asv_radius * (asv_omega**2) * np.cos(asv_omega * t),
            -asv_radius * (asv_omega**2) * np.sin(asv_omega * t),
            np.zeros_like(t),
        ],
        axis=1,
    )

    # Yaw aligned with velocity direction (tangent to circle)
    asv_yaw = np.array([_ned_yaw_from_velocity(v, fallback_yaw=0.0) for v in asv_vel])

    # Build ASV states (biases = 0 ground truth)
    asv_states = []
    for i, ti in enumerate(t):
        asv_states.append(
            (
                float(ti),
                AsvNominalState(
                    pos=asv_pos[i],
                    vel=asv_vel[i],
                    ori=RotationQuaterion.from_euler([0.0, 0.0, float(asv_yaw[i])]),
                    accm_bias=np.zeros(3),
                    gyro_bias=np.zeros(3),
                ),
            )
        )
    # -------------------------------------------------------------------------
    # ASV ideal IMU: body angular rates + specific force in body frame
    # -------------------------------------------------------------------------
    # Convention consistent with your ModelIMU.predict_nom:
    #   acc_world = R(q) @ acc_body + g
    # => acc_body = R(q)^T @ (acc_world - g)
    #
    # Here, acc_world is asv_acc (NED), g is [0,0,9.82] (down positive in NED).
    g = np.array([0.0, 0.0, 9.82])

    imu_meas = []
    for i, ti in enumerate(t):
        q = asv_states[i][1].ori
        R_nb = q.as_rotmat()          # likely nav-from-body, based on your ModelIMU usage
        R_bn = R_nb.T

        # specific force in body
        f_b = R_bn @ (asv_acc[i] - g)

        # body angular rate:
        # since roll=pitch=0 and yaw changes smoothly, omega_b approx [0,0,yaw_rate]
        # yaw_rate is constant here = omega (for perfect circle), in NED.
        # For small roll/pitch, omega_body ≈ [0,0, yaw_rate]
        omega_b = np.array([0.0, 0.0, asv_omega])

        imu_meas.append((float(ti), ImuMeasurement(acc=f_b, avel=omega_b)))
        
    return TimeSequence(asv_states), TimeSequence(imu_meas)

def asv_diamond_traj(t: np.ndarray, duration: float = 300.0, dt: float = 0.1):
    # ASV trajectory to maximize bearing observability: diamond pattern around ROV, with corners at (20,0), (0,20), (-20,0), (0,-20) in NED.
    # Speed oscillates to improve bearing observability?
    return NotImplementedError("TODO")

def asv_circle_var_speed_traj(t: np.ndarray, duration: float = 300.0, dt: float = 0.1):
    # ASV trajectory to maximize bearing observability: circle around ROV, but with varying speed to improve observability
    asv_center = np.array([30.0, 0.0, 0.0])
    asv_radius = 25.0
    asv_omega_base = 0.05   # mean angular velocity, rad/s
    asv_omega_amp  = 0.02   # oscillation amplitude, rad/s  (~40 % of base)
    asv_omega_mod  = 0.10   # oscillation frequency, rad/s  (~63 s period)

    # Instantaneous angular velocity: omega(t) = omega_base + omega_amp*sin(omega_mod*t)
    asv_omega_t = asv_omega_base + asv_omega_amp * np.sin(asv_omega_mod * t)

    # Angular acceleration: d(omega)/dt = omega_amp * omega_mod * cos(omega_mod*t)
    asv_alpha_t = asv_omega_amp * asv_omega_mod * np.cos(asv_omega_mod * t)

    # Heading angle: integral of omega(t)
    # theta(t) = omega_base*t - (omega_amp/omega_mod)*cos(omega_mod*t) + C
    # Choose C so theta(0)=0: C = omega_amp/omega_mod
    asv_theta = (
        asv_omega_base * t
        - (asv_omega_amp / asv_omega_mod) * np.cos(asv_omega_mod * t)
        + (asv_omega_amp / asv_omega_mod)
    )

    # Position in NED
    asv_pos = np.stack(
        [
            asv_center[0] + asv_radius * np.cos(asv_theta),
            asv_center[1] + asv_radius * np.sin(asv_theta),
            np.zeros_like(t),
        ],
        axis=1,
    )

    # Velocity in NED: d/dt [R cos(theta), R sin(theta)] = R * omega(t) * [-sin, cos]
    asv_vel = np.stack(
        [
            -asv_radius * asv_omega_t * np.sin(asv_theta),
            +asv_radius * asv_omega_t * np.cos(asv_theta),
            np.zeros_like(t),
        ],
        axis=1,
    )

    # Acceleration in NED: centripetal + tangential
    asv_acc = np.stack(
        [
            -asv_radius * asv_omega_t**2 * np.cos(asv_theta)
            - asv_radius * asv_alpha_t  * np.sin(asv_theta),
            -asv_radius * asv_omega_t**2 * np.sin(asv_theta)
            + asv_radius * asv_alpha_t  * np.cos(asv_theta),
            np.zeros_like(t),
        ],
        axis=1,
    )

    # Yaw aligned with velocity direction (tangent to circle): theta + pi/2
    asv_yaw = asv_theta + np.pi / 2

    # Build ASV states (biases = 0 ground truth)
    asv_states = []
    for i, ti in enumerate(t):
        asv_states.append(
            (
                float(ti),
                AsvNominalState(
                    pos=asv_pos[i],
                    vel=asv_vel[i],
                    ori=RotationQuaterion.from_euler([0.0, 0.0, float(asv_yaw[i])]),
                    accm_bias=np.zeros(3),
                    gyro_bias=np.zeros(3),
                ),
            )
        )

    # -------------------------------------------------------------------------
    # ASV ideal IMU: body angular rates + specific force in body frame
    # -------------------------------------------------------------------------
    # Convention consistent with your ModelIMU.predict_nom:
    #   acc_world = R(q) @ acc_body + g
    # => acc_body = R(q)^T @ (acc_world - g)
    #
    # Here, acc_world is asv_acc (NED), g is [0,0,9.82] (down positive in NED).
    g = np.array([0.0, 0.0, 9.82])

    imu_meas = []
    for i, ti in enumerate(t):
        q = asv_states[i][1].ori
        R_nb = q.as_rotmat()          # likely nav-from-body, based on your ModelIMU usage
        R_bn = R_nb.T

        # specific force in body
        f_b = R_bn @ (asv_acc[i] - g)

        # body angular rate:
        # since roll=pitch=0 and yaw changes smoothly, omega_b approx [0,0,yaw_rate]
        # yaw_rate is constant here = omega (for perfect circle), in NED.
        # For small roll/pitch, omega_body ≈ [0,0, yaw_rate]
        omega_b = np.array([0.0, 0.0, asv_omega_t[i]])

        imu_meas.append((float(ti), ImuMeasurement(acc=f_b, avel=omega_b)))
    
    return TimeSequence(asv_states), TimeSequence(imu_meas)

def rov_waypoint_spline_traj(t: np.ndarray, duration: float = 300.0, dt: float = 0.1):
    # ROV trajectory to improve observability: figure-8 pattern in NED, centered
    waypoints = [
        (np.array([0.0, 0.0, 5.0]), 0.0),
        (np.array([20.0, 5.0, 10.0]), 60.0),
        (np.array([40.0, 0.0, 15.0]), 120.0),
        (np.array([40.0, 20.0, 20.0]), 180.0),
        (np.array([20.0, 20.0, 12.0]), 240.0),
        (np.array([0.0, 0.0, 5.0]), 300.0),
    ]

    rov_states = []
    for i in range(len(waypoints) - 1):
        p0, t0 = waypoints[i]
        p1, t1 = waypoints[i + 1]
        seg_vel = (p1 - p0) / (t1 - t0)

        seg_t = t[(t >= t0) & (t < t1)]
        for ti in seg_t:
            alpha = (ti - t0) / (t1 - t0)
            pos = p0 + alpha * (p1 - p0)

            rov_states.append(
                (
                    float(ti),
                    RovNominalCV(
                        pos=pos,
                        vel=seg_vel,
                    ),
                )
            )
    return TimeSequence(rov_states)

def rov_fig8_traj(t: np.ndarray, duration: float = 300.0, dt: float = 0.1):
    # ROV trajectory to improve observability: smooth spline through waypoints in NED, with constant speed
    waypoints_new = [
        np.array([0.0, 0.0, 5.0]),
        np.array([10.0, 20.0, 10.0]),
        np.array([20.0, 10.0, 15.0]),
        np.array([30.0, 20.0, 20.0]),
        np.array([40.0, 0.0, 25.0]),
        np.array([30.0, -20.0, 18.0]),
        np.array([20.0, -10.0, 15.0]),
        np.array([10.0, -20.0, 12.0]),
        np.array([0.0, 0.0, 5.0]),
    ]

    pts = np.array(waypoints_new)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Build spline

    tck, u = splprep([x, y, z], s=0, k=3)

    # High‑res sampling for arc length
    u_fine = np.linspace(0, 1, 5000)
    x_f, y_f, z_f = splev(u_fine, tck)

    diffs = np.sqrt(np.diff(x_f)**2 + np.diff(y_f)**2 + np.diff(z_f)**2)
    arc_length = np.concatenate(([0], np.cumsum(diffs)))
    arc_length_norm = arc_length / arc_length[-1]

    # Constant‑speed parameterization for the simulation times
    t_norm = np.zeros_like(t)
    if t[-1] > t[0]:
           t_norm = (t - t[0]) / (t[-1] - t[0])
    u_const = np.interp(t_norm, arc_length_norm, u_fine)

    # Evaluate spline at constant speed
    x_c, y_c, z_c = splev(u_const, tck)

    # Compute velocity by differentiating spline
    dx_du, dy_du, dz_du = splev(u_const, tck, der=1)
    du_dt = np.gradient(u_const, t, edge_order=2)
    dx = dx_du * du_dt
    dy = dy_du * du_dt
    dz = dz_du * du_dt

    rov_states = []
    for i, ti in enumerate(t):
        pos = np.array([x_c[i], y_c[i], z_c[i]])
        vel = np.array([dx[i], dy[i], dz[i]])

        rov_states.append(
            (
                float(ti),
                RovNominalCV(
                    pos=pos,
                    vel=vel,
                ),
            )
        )
    return TimeSequence(rov_states)