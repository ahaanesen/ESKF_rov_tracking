import numpy as np
from asv_states import ASVState, RangeMeasurement, UsblMeasurement
from rov_states import DepthMeasurement, ImuMeasurement, NominalState
from senfuslib import TimeSequence


class MeasurementGenerator:
    def __init__(self,
                 asv_tseq: TimeSequence[ASVState],
                 rov_tseq: TimeSequence[NominalState]):
        self.asv_tseq = asv_tseq
        self.rov_tseq = rov_tseq

    def generate_usbl(self, std_rad: float, lever_arm: np.ndarray,
                      rate_hz: float = 1.0) -> TimeSequence:
        """Generates USBL azimuth/elevation measurements in ASV body frame."""
        t_start = max(self.asv_tseq.t_min, self.rov_tseq.t_min)
        t_end   = min(self.asv_tseq.t_max, self.rov_tseq.t_max)
        times = np.arange(t_start, t_end, 1.0 / rate_hz)

        z_list = []
        for t in times:
            asv = self.asv_tseq.at_time(t)
            rov = self.rov_tseq.at_time(t)

            p_rel_w = rov.pos - asv.pos
            R_wb = asv.ori.as_rotmat()
            p_rel_b = R_wb.T @ p_rel_w - lever_arm

            azi  = np.arctan2(p_rel_b[1], p_rel_b[0])
            #azi = azi % (2 * np.pi) # Ensure azimuth is in [0, 2*pi)
            elev = np.arctan2(-p_rel_b[2], np.linalg.norm(p_rel_b[:2])) # Positive elevation is downwards in NED frame TODO: Check sign convention

            noise = np.random.normal(0, std_rad, 2)
            #z_list.append((t, UsblMeasurement(np.array([azi + noise[0], elev + noise[1]]))))
            z_list.append((t, UsblMeasurement.from_array(
                np.array([azi % (2 * np.pi), -elev]) + noise  # wrap azi, negate elev for NED convention
            )))
        return TimeSequence(z_list)

    def generate_range(self, std_m: float, lever_arm: np.ndarray,
                       rate_hz: float = 1.0) -> TimeSequence:
        """Generates Euclidean distance measurements from ASV sensor to ROV."""
        t_start = max(self.asv_tseq.t_min, self.rov_tseq.t_min)
        t_end   = min(self.asv_tseq.t_max, self.rov_tseq.t_max)
        times = np.arange(t_start, t_end, 1.0 / rate_hz)

        z_list = []
        for t in times:
            asv = self.asv_tseq.at_time(t)
            rov = self.rov_tseq.at_time(t)

            sensor_pos = asv.pos + asv.ori.as_rotmat() @ lever_arm
            dist = np.linalg.norm(sensor_pos - rov.pos)
        z_list.append((t, RangeMeasurement.from_array(np.array([dist + np.random.normal(0, std_m)]))))

        return TimeSequence(z_list)

    def generate_depth(self, std_m: float,
                       rate_hz: float = 5.0) -> TimeSequence:
        """Generates depth (NED Down) measurements."""
        t_start = self.rov_tseq.t_min
        t_end   = self.rov_tseq.t_max
        times = np.arange(t_start, t_end, 1.0 / rate_hz)

        z_list = []
        for t in times:
            rov = self.rov_tseq.at_time(t)
            z_list.append((t, DepthMeasurement(np.array([rov.pos[2] + np.random.normal(0, std_m)]))))

        return TimeSequence(z_list)

    # def generate_imu(self, accm_std: float, gyro_std: float,
    #                  rate_hz: float = 100.0) -> TimeSequence:
    #     """
    #     Generates IMU measurements from ROV ground truth.
    #     Accelerometer: specific force in body frame (accounts for gravity).
    #     Gyroscope: angular velocity from finite-differenced quaternions.
    #     """
    #     g_ned = np.array([0.0, 0.0, 9.81])  # gravity in NED (positive down)
    #     t_start = self.rov_tseq.t_min
    #     t_end   = self.rov_tseq.t_max
    #     dt = 1.0 / rate_hz
    #     times = np.arange(t_start, t_end, dt)

    #     z_list = []
    #     for i, t in enumerate(times):
    #         r_curr = self.rov_tseq.at_time(t)
    #         R_wb = r_curr.ori.as_rotmat()

    #         # --- Accelerometer: specific force = a_body - R^T @ g ---
    #         if i == 0 or i == len(times) - 1:
    #             acc_w = np.zeros(3)
    #         else:
    #             r_prev = self.rov_tseq.at_time(times[i - 1])
    #             r_next = self.rov_tseq.at_time(times[i + 1])
    #             acc_w = (r_next.vel - r_prev.vel) / (2 * dt)

    #         specific_force_b = R_wb.T @ (acc_w - g_ned)
    #         acc_noisy = specific_force_b + np.random.normal(0, accm_std, 3)

    #         # --- Gyroscope: omega from finite-differenced rotation matrices ---
    #         if i == 0 or i == len(times) - 1:
    #             omega_b = np.zeros(3)
    #         else:
    #             r_prev = self.rov_tseq.at_time(times[i - 1])
    #             r_next = self.rov_tseq.at_time(times[i + 1])
    #             R_prev = r_prev.ori.as_rotmat()
    #             R_next = r_next.ori.as_rotmat()
    #             # Skew-symmetric omega matrix in body frame
    #             dR = R_wb.T @ (R_next - R_prev) / (2 * dt)
    #             omega_b = np.array([dR[2, 1], dR[0, 2], dR[1, 0]])

    #         gyro_noisy = omega_b + np.random.normal(0, gyro_std, 3)
    #         z_list.append((t, ImuMeasurement(acc_noisy, gyro_noisy)))

    #     return TimeSequence(z_list)