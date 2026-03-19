from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from senfuslib import TimeSequence

from tracking_and_navigation.states import JointEskfState, JointNominalState, AsvNominalState, RovNominalCV
from tracking_and_navigation.measurements import (
    GnssMeasurement,
    UsblMeasurement,
    RangeMeasurement,
    DepthMeasurement,
)

mpl.rcParams["axes.grid"] = True
mpl.rcParams["legend.loc"] = "lower right"
mpl.rcParams["legend.fontsize"] = "small"


def _extract_pos(tseq: TimeSequence, getter) -> np.ndarray:
    return np.stack([getter(v) for v in tseq.values])


@dataclass
class PlotterESKFJoint:
    # Ground truth and estimates
    rov_gt: TimeSequence[RovNominalCV]          # ROV ground truth (CV)
    asv_gt: TimeSequence[AsvNominalState]       # ASV ground truth (nominal with pos/vel/ori/bias)

    x_upds: TimeSequence[JointEskfState]        # joint updated states
    x_preds: TimeSequence[JointEskfState]       # joint predicted states

    # Measurements (optional)
    z_gnss_asv: TimeSequence[GnssMeasurement] = None
    z_usbl:     TimeSequence[UsblMeasurement] = None
    z_range:    TimeSequence[RangeMeasurement] = None
    z_depth:    TimeSequence[DepthMeasurement] = None

    scenario_name: str = "Joint scenario"
    save_dir: str = None

    # ------------------------
    # Helpers: extract arrays
    # ------------------------
    def _rov_est_pos(self, tseq: TimeSequence[JointEskfState]) -> np.ndarray:
        return np.stack([v.nom.rov.pos for v in tseq.values])

    def _rov_est_vel(self, tseq: TimeSequence[JointEskfState]) -> np.ndarray:
        return np.stack([v.nom.rov.vel for v in tseq.values])

    def _asv_est_pos(self, tseq: TimeSequence[JointEskfState]) -> np.ndarray:
        return np.stack([v.nom.asv.pos for v in tseq.values])

    def _asv_est_vel(self, tseq: TimeSequence[JointEskfState]) -> np.ndarray:
        return np.stack([v.nom.asv.vel for v in tseq.values])

    def _rov_est_std(self, tseq: TimeSequence[JointEskfState]) -> np.ndarray:
        """
        3-sigma std on ROV position from joint covariance.
        Joint error layout (from your JointIdx):
          ASV: 0..14
          ROV: 15..20, where ROV pos is 15..17
        """
        stds = []
        for v in tseq.values:
            P = v.err.cov
            rov_pos_var = np.diag(P)[15:18]
            stds.append(3.0 * np.sqrt(rov_pos_var))
        return np.stack(stds)

    def _asv_est_std(self, tseq: TimeSequence[JointEskfState]) -> np.ndarray:
        """3-sigma std on ASV position from joint covariance (ASV pos is 0..2)."""
        stds = []
        for v in tseq.values:
            P = v.err.cov
            asv_pos_var = np.diag(P)[0:3]
            stds.append(3.0 * np.sqrt(asv_pos_var))
        return np.stack(stds)

    # ------------------------
    # Plots
    # ------------------------
    def plot3d(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ned_sign = np.array([1, 1, -1])  # flip z for plotting (up positive)

        # ROV GT
        if self.rov_gt is not None:
            gt_pos = _extract_pos(self.rov_gt, attrgetter("pos")) * ned_sign
            ax.plot(*gt_pos.T, label="ROV ground truth", linestyle="--", color="C1", alpha=0.8)
            ax.scatter(*gt_pos[0], marker="x", color="red", s=60, label="ROV start")

        # ROV estimate
        if self.x_upds is not None and self.x_upds.values:
            est_pos = self._rov_est_pos(self.x_upds) * ned_sign
            ax.plot(*est_pos.T, label="ROV estimate (upd)", color="C0", alpha=0.8)

        # ASV GT
        if self.asv_gt is not None:
            asv_pos = _extract_pos(self.asv_gt, attrgetter("pos")) * ned_sign
            ax.plot(*asv_pos.T, label="ASV ground truth", color="C2", linestyle="-.", alpha=0.6)
            ax.scatter(*asv_pos[0], marker="^", color="C2", s=60)

        # ASV estimate
        if self.x_upds is not None and self.x_upds.values:
            asv_est_pos = self._asv_est_pos(self.x_upds) * ned_sign
            ax.plot(*asv_est_pos.T, label="ASV estimate (upd)", color="C3", alpha=0.8)

        ax.set_xlabel("North [m]")
        ax.set_ylabel("East [m]")
        ax.set_zlabel("Up [m]")  # we flipped sign
        ax.set_title(f"{self.scenario_name} — 3D Trajectories")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_rov_position(self):
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        labels = ["North [m]", "East [m]", "Down [m]"]

        times_upd = np.array(self.x_upds.times)
        est_pos = self._rov_est_pos(self.x_upds)
        std3 = self._rov_est_std(self.x_upds)

        for i, (ax, lbl) in enumerate(zip(axs, labels)):
            # GT
            if self.rov_gt is not None:
                gt_times = np.array(self.rov_gt.times)
                gt_pos = _extract_pos(self.rov_gt, attrgetter("pos"))
                ax.plot(gt_times, gt_pos[:, i], label="GT", linestyle="--", color="C1", alpha=0.8)

            # Est + band
            ax.plot(times_upd, est_pos[:, i], label="Est", color="C0")
            ax.fill_between(
                times_upd,
                est_pos[:, i] - std3[:, i],
                est_pos[:, i] + std3[:, i],
                alpha=0.2,
                color="C0",
                label="±3σ",
            )

            # Depth meas only on Down
            if i == 2 and self.z_depth is not None:
                depth_times = np.array(self.z_depth.times)
                depth_vals = np.array([float(v[0]) for v in self.z_depth.values])
                ax.scatter(depth_times, depth_vals, s=8, color="C3", alpha=0.5, label="Depth meas", zorder=5)

            ax.set_ylabel(lbl)
            ax.legend()

        axs[0].set_title(f"{self.scenario_name} — ROV Position")
        axs[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        return fig

    def plot_asv_position(self):
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        labels = ["North [m]", "East [m]", "Down [m]"]

        times_upd = np.array(self.x_upds.times)
        est_pos = self._asv_est_pos(self.x_upds)
        std3 = self._asv_est_std(self.x_upds)

        # Precompute GNSS arrays once (correctly)
        if self.z_gnss_asv is not None:
            gnss_times = np.array(self.z_gnss_asv.times)
            gnss_pos = np.stack([np.asarray(m.pos, dtype=float).reshape(3) for m in self.z_gnss_asv.values])  # (N,3)
        else:
            gnss_times, gnss_pos = None, None

        for i, (ax, lbl) in enumerate(zip(axs, labels)):
            if self.asv_gt is not None:
                gt_times = np.array(self.asv_gt.times)
                gt_pos = _extract_pos(self.asv_gt, attrgetter("pos"))
                ax.plot(gt_times, gt_pos[:, i], label="GT", linestyle="--", color="C2", alpha=0.8)

            
            if gnss_pos is not None:
                ax.scatter(
                    gnss_times,
                    gnss_pos[:, i],
                    s=10,
                    color="C4",
                    alpha=0.4,
                    label="GNSS meas" if i == 0 else None,  # avoid repeated legend entries
                    zorder=5,
                )
            # if self.z_gnss_asv is not None:
            #     gnss_times = np.array(self.z_gnss_asv.times)
            #     gnss_pos = np.array([float(v[0]) for v in self.z_gnss_asv.values]) # (N,3)
            #     ax.scatter(gnss_times, gnss_pos, s=10, color="C4", alpha=0.4, label="GNSS meas") #  zorder=5

            ax.plot(times_upd, est_pos[:, i], label="Est", color="C3")
            ax.fill_between(
                times_upd,
                est_pos[:, i] - std3[:, i],
                est_pos[:, i] + std3[:, i],
                alpha=0.2,
                color="C3",
                label="±3σ",
            )

            ax.set_ylabel(lbl)
            ax.legend()

        axs[0].set_title(f"{self.scenario_name} — ASV Position")
        axs[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        return fig

    def plot_usbl_measurements(self):
        if self.z_usbl is None:
            return None
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        times = np.array(self.z_usbl.times)
        azi = np.array([float(v[0]) for v in self.z_usbl.values])
        elev = np.array([float(v[1]) for v in self.z_usbl.values])

        axs[0].plot(times, np.rad2deg(azi), ".", color="C4", markersize=4)
        axs[0].set_ylabel("Azimuth [deg]")
        axs[1].plot(times, np.rad2deg(elev), ".", color="C5", markersize=4)
        axs[1].set_ylabel("Elevation [deg]")
        axs[0].set_title(f"{self.scenario_name} — USBL Measurements")
        axs[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        return fig

    def plot_range_measurements(self):
        if self.z_range is None:
            return None
        fig, ax = plt.subplots(figsize=(10, 3))
        times = np.array(self.z_range.times)
        ranges = np.array([float(v[0]) for v in self.z_range.values])

        # True range for reference (use lever arm = 0 here unless you want to include it)
        if self.rov_gt is not None and self.asv_gt is not None:
            true_ranges = []
            for t in times:
                rov = self.rov_gt.at_time(t)
                asv = self.asv_gt.at_time(t)
                true_ranges.append(np.linalg.norm(np.asarray(rov.pos) - np.asarray(asv.pos)))
            ax.plot(times, true_ranges, label="True (ASV pos to ROV pos)", linestyle="--", color="C1", alpha=0.8)

        ax.plot(times, ranges, ".", color="C4", markersize=4, label="Measured")
        ax.set_ylabel("Range [m]")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"{self.scenario_name} — Range Measurements")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_rov_position_error(self):
        """ROV position error (est - gt) over time with ±3σ from joint covariance."""
        if self.rov_gt is None:
            return None

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
        labels = ["err N [m]", "err E [m]", "err D [m]"]

        times_upd = np.array(self.x_upds.times)
        est_pos = self._rov_est_pos(self.x_upds)
        std3 = self._rov_est_std(self.x_upds)

        gt_pos_at_upd = np.stack([self.rov_gt.at_time(t).pos for t in times_upd])
        error = est_pos - gt_pos_at_upd

        for i, (ax, lbl) in enumerate(zip(axs, labels)):
            ax.plot(times_upd, error[:, i], color="C0", label="error")
            ax.fill_between(times_upd, -std3[:, i], std3[:, i], alpha=0.2, color="C0", label="±3σ")
            ax.axhline(0.0, color="k", linewidth=0.5)
            ax.set_ylabel(lbl)
            ax.legend()

        axs[0].set_title(f"{self.scenario_name} — ROV Position Error")
        axs[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        return fig

    def show(self):
        self._save(self.plot3d(), "3d_trajectory")
        self._save(self.plot_rov_position(), "rov_position")
        self._save(self.plot_asv_position(), "asv_position")
        self._save(self.plot_rov_position_error(), "rov_position_error")
        self._save(self.plot_usbl_measurements(), "usbl")
        self._save(self.plot_range_measurements(), "range")
        plt.show(block=True)

    def _save(self, fig, name: str):
        if fig is None:
            return
        if self.save_dir is None:
            return
        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f"{name}.png", dpi=150, bbox_inches="tight")