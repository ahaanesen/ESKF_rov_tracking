import numpy as np

from quaternion import RotationQuaterion
from senfuslib.timesequence import TimeSequence

from tracking_and_navigation.states import ASVNominalState, ROVNominalCV
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
    # -------------------------------------------------------------------------
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
                ASVNominalState(
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

    # -------------------------------------------------------------------------
    # ROV ground truth: waypoint path with depth changes (CV nominal)
    # -------------------------------------------------------------------------
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
                    ROVNominalCV(
                        pos=pos,
                        vel=seg_vel,
                    ),
                )
            )

    return TimeSequence(asv_states), TimeSequence(rov_states), TimeSequence(imu_meas)