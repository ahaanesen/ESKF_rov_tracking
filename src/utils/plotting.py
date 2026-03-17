from dataclasses import dataclass, field
from operator import attrgetter
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

from senfuslib import TimeSequence, MultiVarGauss
from rov_states import EskfState, NominalState
from asv_states import ASVState, UsblMeasurement, RangeMeasurement
from rov_states import DepthMeasurement


mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.loc'] = 'lower right'
mpl.rcParams['legend.fontsize'] = 'small'


def _extract_pos(tseq: TimeSequence, getter) -> np.ndarray:
    return np.stack([getter(v) for v in tseq.values])


@dataclass
class PlotterESKF:
    # Ground truth and estimates
    rov_gt:      TimeSequence[NominalState]         # ROV ground truth
    asv_gt:      TimeSequence[ASVState]             # ASV ground truth
    rov_upds:    TimeSequence[EskfState]            # ESKF updated states
    rov_preds:   TimeSequence[EskfState]            # ESKF predicted states

    # Measurements (all optional)
    z_usbl:  TimeSequence[UsblMeasurement]  = None
    z_range: TimeSequence[RangeMeasurement] = None
    z_depth: TimeSequence[DepthMeasurement] = None

    scenario_name: str = "Scenario"
    save_dir: str = None # Optional directory to save plots instead of showing them interactively, eg. "plots/"


    def _rov_est_pos(self, tseq: TimeSequence[EskfState]) -> np.ndarray:
        return np.stack([v.nom.pos for v in tseq.values])

    def _rov_est_vel(self, tseq: TimeSequence[EskfState]) -> np.ndarray:
        return np.stack([v.nom.vel for v in tseq.values])

    def _rov_est_std(self, tseq: TimeSequence[EskfState]) -> np.ndarray:
        """Return 3-sigma position std from error covariance."""
        return np.stack([3 * np.sqrt(np.diag(v.err.cov)[:3])
                         for v in tseq.values])

    def plot3d(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ned_sign = np.array([1, 1, -1])  # flip z for plotting (up positive)

        # ROV ground truth
        if self.rov_gt is not None:
            gt_pos = _extract_pos(self.rov_gt, attrgetter('pos')) * ned_sign
            ax.plot(*gt_pos.T, label='ROV ground truth',
                    linestyle='--', color='C1', alpha=0.8)
            ax.scatter(*gt_pos[0], marker='x', color='red', s=60, label='ROV start')

        # ROV estimate
        est_pos = self._rov_est_pos(self.rov_upds) * ned_sign
        ax.plot(*est_pos.T, label='ROV estimate', color='C0', alpha=0.8)

        # ASV ground truth
        if self.asv_gt is not None:
            asv_pos = _extract_pos(self.asv_gt, attrgetter('pos')) * ned_sign
            ax.plot(*asv_pos.T, label='ASV', color='C2',
                    linestyle='-.', alpha=0.6)
            ax.scatter(*asv_pos[0], marker='^', color='C2', s=60)

        ax.set_xlabel('North [m]')
        ax.set_ylabel('East [m]')
        ax.set_zlabel('Down [m]')
        ax.set_title(f'{self.scenario_name} — 3D Trajectories')
        ax.legend()
        fig.tight_layout()

        return fig

    def plot_position(self):
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        labels = ['North [m]', 'East [m]', 'Down [m]']
        times_upd = np.array(self.rov_upds.times)
        est_pos = self._rov_est_pos(self.rov_upds)
        std3 = self._rov_est_std(self.rov_upds)

        for i, (ax, lbl) in enumerate(zip(axs, labels)):
            # Ground truth
            if self.rov_gt is not None:
                gt_pos = _extract_pos(self.rov_gt, attrgetter('pos'))
                gt_times = np.array(self.rov_gt.times)
                ax.plot(gt_times, gt_pos[:, i], label='GT',
                        linestyle='--', color='C1', alpha=0.8)

            # Estimate + 3-sigma band
            ax.plot(times_upd, est_pos[:, i], label='Est', color='C0')
            ax.fill_between(times_upd,
                            est_pos[:, i] - std3[:, i],
                            est_pos[:, i] + std3[:, i],
                            alpha=0.2, color='C0', label='±3σ')

            # Depth measurements on down-axis
            if i == 2 and self.z_depth is not None:
                depth_times = np.array(self.z_depth.times)
                depth_vals = np.array([float(v[0]) for v in self.z_depth.values])
                ax.scatter(depth_times, depth_vals, s=8, color='C3',
                           alpha=0.5, label='Depth meas', zorder=5)

            ax.set_ylabel(lbl)
            ax.legend()

        axs[0].set_title(f'{self.scenario_name} — Position')
        axs[-1].set_xlabel('Time [s]')
        fig.tight_layout()

        return fig

    def plot_velocity(self):
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        labels = ['vN [m/s]', 'vE [m/s]', 'vD [m/s]']
        times_upd = np.array(self.rov_upds.times)
        est_vel = self._rov_est_vel(self.rov_upds)

        for i, (ax, lbl) in enumerate(zip(axs, labels)):
            if self.rov_gt is not None:
                gt_vel = _extract_pos(self.rov_gt, attrgetter('vel'))
                gt_times = np.array(self.rov_gt.times)
                ax.plot(gt_times, gt_vel[:, i], label='GT',
                        linestyle='--', color='C1', alpha=0.8)
            ax.plot(times_upd, est_vel[:, i], label='Est', color='C0')
            ax.set_ylabel(lbl)
            ax.legend()

        axs[0].set_title(f'{self.scenario_name} — Velocity')
        axs[-1].set_xlabel('Time [s]')
        fig.tight_layout()

        return fig

    def plot_usbl_measurements(self):
        if self.z_usbl is None:
            return
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        times = np.array(self.z_usbl.times)
        azi  = np.array([float(v[0]) for v in self.z_usbl.values])
        elev = np.array([float(v[1]) for v in self.z_usbl.values])

        axs[0].plot(times, np.rad2deg(azi),  '.', color='C4', markersize=4)
        axs[0].set_ylabel('Azimuth [deg]')
        axs[1].plot(times, np.rad2deg(elev), '.', color='C5', markersize=4)
        axs[1].set_ylabel('Elevation [deg]')
        axs[0].set_title(f'{self.scenario_name} — USBL Measurements')
        axs[-1].set_xlabel('Time [s]')
        fig.tight_layout()

        return fig

    def plot_range_measurements(self):
        if self.z_range is None:
            return
        fig, ax = plt.subplots(figsize=(10, 3))
        times = np.array(self.z_range.times)
        ranges = np.array([float(v[0]) for v in self.z_range.values])

        # True range from ground truth for comparison
        if self.rov_gt is not None and self.asv_gt is not None:
            true_ranges = []
            for t in times:
                rov = self.rov_gt.at_time(t)
                asv = self.asv_gt.at_time(t)
                true_ranges.append(np.linalg.norm(rov.pos - asv.pos))
            ax.plot(times, true_ranges, label='True range',
                    linestyle='--', color='C1', alpha=0.8)

        ax.plot(times, ranges, '.', color='C4', markersize=4, label='Measured')
        ax.set_ylabel('Range [m]')
        ax.set_xlabel('Time [s]')
        ax.set_title(f'{self.scenario_name} — Range Measurements')
        ax.legend()
        fig.tight_layout()

        return fig

    def plot_position_error(self):
        """Plot position error (est - gt) over time."""
        if self.rov_gt is None:
            return

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
        labels = ['err N [m]', 'err E [m]', 'err D [m]']
        times_upd = np.array(self.rov_upds.times)
        est_pos = self._rov_est_pos(self.rov_upds)
        std3 = self._rov_est_std(self.rov_upds)

        gt_pos_at_upd = np.stack([
            self.rov_gt.at_time(t).pos for t in times_upd
        ])
        error = est_pos - gt_pos_at_upd

        for i, (ax, lbl) in enumerate(zip(axs, labels)):
            ax.plot(times_upd, error[:, i], color='C0', label='error')
            ax.fill_between(times_upd, -std3[:, i], std3[:, i],
                            alpha=0.2, color='C0', label='±3σ')
            ax.axhline(0, color='k', linewidth=0.5)
            ax.set_ylabel(lbl)
            ax.legend()

        axs[0].set_title(f'{self.scenario_name} — Position Error')
        axs[-1].set_xlabel('Time [s]')
        fig.tight_layout()

        return fig

    def show(self):
        fig = self.plot3d();              self._save(fig, "3d_trajectory")
        fig = self.plot_position();       self._save(fig, "position")
        fig = self.plot_velocity();       self._save(fig, "velocity")
        fig = self.plot_position_error(); self._save(fig, "position_error")
        fig = self.plot_usbl_measurements(); self._save(fig, "usbl")
        fig = self.plot_range_measurements(); self._save(fig, "range")
        plt.show(block=True)

    def _save(self, fig, name: str):
        if fig is None:
            return
        if self.save_dir is not None:
            path = Path(self.save_dir)
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(path / f"{name}.png", dpi=150, bbox_inches='tight')
