import numpy as np

from tracking_only.asv_states import ASVState
from quaternion import RotationQuaterion
from tracking_only.rov_states import NominalState
from senfuslib.timesequence import TimeSequence


def generate_trajectories(duration=300, dt=0.1):
    t = np.arange(0, duration, dt)
    
    # --- ASV Trajectory (Orbiting an offset from origin so it is never directly above the ROV) ---
    asv_center = np.array([30, 0, 0])
    asv_radius = 25
    asv_omega = 0.05  # angular velocity
    
    asv_pos = np.stack([
        asv_center[0] + asv_radius * np.cos(asv_omega * t),
        asv_center[1] + asv_radius * np.sin(asv_omega * t),
        np.zeros_like(t) # ASV stays on surface
    ], axis=1)
    
    # Calculate orientation (pointing along velocity vector)
    asv_yaw = asv_omega * t + np.pi/2
    
    asv_states = []
    for i in range(len(t)):
        state = ASVState(
            pos=asv_pos[i],
            ori=RotationQuaterion.from_euler([0, 0, asv_yaw[i]]),
            # vel=np.array([-asv_radius * asv_omega * np.sin(asv_omega * t[i]),
            #               asv_radius * asv_omega * np.cos(asv_omega * t[i]), 
            #               0])
        )
        asv_states.append((t[i], state))
        
    # # --- ROV Trajectory (Descending Spiral) ---
    # rov_radius = 10
    # rov_omega = 0.08
    # rov_descend_rate = 0.1 # m/s
    
    # rov_pos = np.stack([
    #     rov_radius * np.cos(rov_omega * t),
    #     rov_radius * np.sin(rov_omega * t),
    #     5 + rov_descend_rate * t # Starts at 5m depth
    # ], axis=1)
    
    # rov_gt_states = []
    # for i in range(len(t)):
    #     # NominalState matches your ESKF ROV state structure
    #     state = NominalState(
    #         pos=rov_pos[i],
    #         vel=np.array([-rov_radius * rov_omega * np.sin(rov_omega * t[i]),
    #                       rov_radius * rov_omega * np.cos(rov_omega * t[i]),
    #                       rov_descend_rate]),
    #         ori=RotationQuaterion.from_euler([0, 0, rov_omega * t[i]]),
    #         accm_bias=np.zeros(3),
    #         gyro_bias=np.zeros(3)
    #     )
    #     rov_gt_states.append((t[i], state))

    # ROV: waypoint path with depth changes
    waypoints = [
        (np.array([ 0.0,  0.0,  5.0]),   0.0),
        (np.array([20.0,  5.0, 10.0]),   60.0),
        (np.array([40.0,  0.0, 15.0]),  120.0),
        (np.array([40.0, 20.0, 20.0]),  180.0),
        (np.array([20.0, 20.0, 12.0]),  240.0),
        (np.array([ 0.0,  0.0,  5.0]),  300.0),
    ]

    rov_gt_states = []
    for i in range(len(waypoints) - 1):
        p0, t0 = waypoints[i]
        p1, t1 = waypoints[i + 1]
        seg_vel = (p1 - p0) / (t1 - t0)
        seg_t = t[(t >= t0) & (t < t1)]
        for ti in seg_t:
            alpha = (ti - t0) / (t1 - t0)
            pos = p0 + alpha * (p1 - p0)
            yaw = np.arctan2(seg_vel[1], seg_vel[0])
            ori = RotationQuaterion.from_euler([0, 0, yaw])
            state = NominalState(pos=pos, vel=seg_vel, ori=ori,
                                 accm_bias=np.zeros(3), gyro_bias=np.zeros(3))
            rov_gt_states.append((ti, state))

        
    return TimeSequence(asv_states), TimeSequence(rov_gt_states)