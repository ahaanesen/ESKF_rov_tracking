import numpy as np

from utils.angles import wrap_to_2pi
from senfuslib import TimeSequence

from tracking_and_navigation.states import AsvNominalState, RovNominalCV
from tracking_and_navigation.measurements import (
    GnssMeasurement,
    ImuMeasurement,
    UsblMeasurement,
    RangeMeasurement,
    DepthMeasurement,
)


class MeasurementGenerator:
    def __init__(
        self,
        asv_tseq: TimeSequence[AsvNominalState],
        rov_tseq: TimeSequence[RovNominalCV],
    ):
        self.asv_tseq = asv_tseq
        self.rov_tseq = rov_tseq

    def generate_usbl(
        self,
        std_rad: float,
        lever_arm: np.ndarray,
        rate_hz: float = 1.0,
    ) -> TimeSequence:
        """USBL azimuth/elevation from ASV-mounted sensor to ROV, all in NED."""
        t_start = max(self.asv_tseq.t_min, self.rov_tseq.t_min)
        t_end = min(self.asv_tseq.t_max, self.rov_tseq.t_max)
        times = np.arange(t_start, t_end, 1.0 / rate_hz)

        z_list = []
        for t in times:
            asv = self.asv_tseq.at_time(t)
            rov = self.rov_tseq.at_time(t)

            sensor_pos = asv.pos + asv.ori.as_rotmat() @ lever_arm
            d = rov.pos - sensor_pos
            dx, dy, dz = d
            r = np.sqrt(dx**2 + dy**2)

            azi = wrap_to_2pi(np.arctan2(dy, dx))
            elev = np.arctan2(dz, r)  # positive down

            noise = np.random.normal(0.0, std_rad, 2)
            z = np.array([azi, elev]) + noise
            z[0] = wrap_to_2pi(z[0])

            z_list.append((float(t), UsblMeasurement.from_array(z)))

        return TimeSequence(z_list)

    def generate_range(
        self,
        std_m: float,
        lever_arm: np.ndarray,
        rate_hz: float = 1.0,
    ) -> TimeSequence:
        """Euclidean distance from ASV sensor (lever arm) to ROV."""
        t_start = max(self.asv_tseq.t_min, self.rov_tseq.t_min)
        t_end = min(self.asv_tseq.t_max, self.rov_tseq.t_max)
        times = np.arange(t_start, t_end, 1.0 / rate_hz)

        z_list = []
        for t in times:
            asv = self.asv_tseq.at_time(t)
            rov = self.rov_tseq.at_time(t)

            sensor_pos = asv.pos + asv.ori.as_rotmat() @ lever_arm
            dist = float(np.linalg.norm(rov.pos - sensor_pos))
            dist_noisy = dist + float(np.random.normal(0.0, std_m))

            z_list.append((float(t), RangeMeasurement.from_array(np.array([dist_noisy]))))

        return TimeSequence(z_list)

    def generate_depth(self, std_m: float, rate_hz: float = 5.0) -> TimeSequence:
        """ROV depth measurement (NED Down = rov.pos[2])."""
        t_start = self.rov_tseq.t_min
        t_end = self.rov_tseq.t_max
        times = np.arange(t_start, t_end, 1.0 / rate_hz)

        z_list = []
        for t in times:
            rov = self.rov_tseq.at_time(t)
            depth = float(rov.pos[2])
            depth_noisy = depth + float(np.random.normal(0.0, std_m))
            z_list.append((float(t), DepthMeasurement.from_array(np.array([depth_noisy]))))

        return TimeSequence(z_list)

    def generate_imu_asv(
        self,
        accm_std: float,
        gyro_std: float,
        rate_hz: float = 100.0,
        g_ned: np.ndarray = np.array([0.0, 0.0, 9.82]),
    ) -> TimeSequence:
        """
        Generate IMU for the ASV ground truth.

        Consistent with ModelIMU.predict_nom():
            acc_world = R_nb @ acc_body + g_ned
        =>  acc_body = R_nb.T @ (acc_world - g_ned)

        Gyro: estimate yaw rate from rotation matrices, assume roll/pitch ~ 0 so
              omega_body ≈ [0, 0, yaw_rate].
        """
        t_start = self.asv_tseq.t_min
        t_end = self.asv_tseq.t_max
        dt = 1.0 / rate_hz
        times = np.arange(t_start, t_end, dt)

        def yaw_from_Rnb(R_nb: np.ndarray) -> float:
            # Works for the common yaw convention when roll/pitch are small.
            return float(np.arctan2(R_nb[1, 0], R_nb[0, 0]))

        z_list = []
        for i, t in enumerate(times):
            s = self.asv_tseq.at_time(t)
            R_nb = s.ori.as_rotmat()

            # --- world-frame acceleration from finite difference of vel ---
            if i == 0 or i == len(times) - 1:
                acc_w = np.zeros(3)
            else:
                s_prev = self.asv_tseq.at_time(times[i - 1])
                s_next = self.asv_tseq.at_time(times[i + 1])
                acc_w = (np.asarray(s_next.vel) - np.asarray(s_prev.vel)) / (2.0 * dt)

            # --- accelerometer specific force in body ---
            f_b = R_nb.T @ (acc_w - g_ned)
            acc_noisy = f_b + np.random.normal(0.0, accm_std, 3)

            # --- gyro from yaw rate (roll/pitch assumed small) ---
            if i == 0 or i == len(times) - 1:
                omega_b = np.zeros(3)
            else:
                R_prev = self.asv_tseq.at_time(times[i - 1]).ori.as_rotmat()
                R_next = self.asv_tseq.at_time(times[i + 1]).ori.as_rotmat()
                yaw_prev = yaw_from_Rnb(R_prev)
                yaw_next = yaw_from_Rnb(R_next)

                # Wrap delta yaw to [-pi, pi] to avoid discontinuity
                dyaw = (yaw_next - yaw_prev + np.pi) % (2.0 * np.pi) - np.pi
                yaw_rate = dyaw / (2.0 * dt)

                omega_b = np.array([0.0, 0.0, yaw_rate])

            gyro_noisy = omega_b + np.random.normal(0.0, gyro_std, 3)

            z_list.append((float(t), ImuMeasurement(acc=acc_noisy, avel=gyro_noisy)))

        return TimeSequence(z_list)
    
    def generate_gnss_asv(
        self,
        std_ne: float,
        std_d: float,
        lever_arm: np.ndarray,
        rate_hz: float = 1.0,
    ) -> TimeSequence:
        """
        Generate ASV GNSS position measurements in NED with lever-arm:
            z = p_asv + R_asv*lever_arm + noise

        Noise:
          north/east ~ N(0, std_ne)
          down       ~ N(0, std_d)
        """
        t_start = self.asv_tseq.t_min
        t_end = self.asv_tseq.t_max
        times = np.arange(t_start, t_end, 1.0 / rate_hz)

        z_list = []
        for t in times:
            asv = self.asv_tseq.at_time(t)

            antenna_pos = asv.pos + asv.ori.as_rotmat() @ lever_arm
            noise = np.array(
                [
                    np.random.normal(0.0, std_ne),
                    np.random.normal(0.0, std_ne),
                    np.random.normal(0.0, std_d),
                ]
            )
            z = np.asarray(antenna_pos) + noise
            z_list.append((float(t), GnssMeasurement.from_array(z)))
        
        return TimeSequence(z_list)