from dataclasses import dataclass, field
from dataclasses import dataclass, field
from operator import attrgetter
from pathlib import Path
import csv
import csv

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2
from scipy.stats import chi2

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)

from senfuslib import TimeSequence

from tracking_and_navigation.states import (
    JointEskfState,
    JointNominalState,
    JointIdx,
    AsvNominalState,
    RovNominalCV,
)
from tracking_and_navigation.measurements import (
    GnssMeasurement,
    UsblMeasurement,
    RangeMeasurement,
    DepthMeasurement,
)
from utils.angles import wrap_to_pi
from utils.angles import wrap_to_pi

mpl.rcParams["axes.grid"] = True
mpl.rcParams["legend.loc"] = "lower right"
mpl.rcParams["legend.fontsize"] = "small"


def _extract_pos(tseq: TimeSequence, getter) -> np.ndarray:
    return np.stack([getter(v) for v in tseq.values])


def _interp(t_src: np.ndarray, x_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """
    Interpolate vector-valued time series.
    t_src: (N,)
    x_src: (N, d)
    t_tgt: (M,)
    returns: (M, d)
    """
    x_i = np.zeros((len(t_tgt), x_src.shape[1]))
    for k in range(x_src.shape[1]):
        x_i[:, k] = np.interp(t_tgt, t_src, x_src[:, k])
    return x_i


def _error_norm(gt: np.ndarray, est: np.ndarray):
    """
    Per-time-step Euclidean error norm across components.
    gt, est: (N, d)
    returns:
      norm: (N,)
      err:  (N, d)
    """
    err = est - gt
    norm = np.linalg.norm(err, axis=1)
    return norm, err


def _path_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return 0.0
    diffs = np.diff(xyz, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _error_stats_from_errvec(err3: np.ndarray) -> dict:
    """
    err3: (N,3) component error
    """
    if len(err3) == 0:
        return {
            "num_samples": 0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "final_error": np.nan,
            "ate_rms": np.nan,
            "mean_abs_n": np.nan,
            "mean_abs_e": np.nan,
            "mean_abs_d": np.nan,
            "std_n": np.nan,
            "std_e": np.nan,
            "std_d": np.nan,
        }

    dist = np.linalg.norm(err3, axis=1)
    return {
        "num_samples": int(len(dist)),
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(np.max(dist)),
        "final_error": float(dist[-1]),
        "ate_rms": float(np.sqrt(np.mean(dist**2))),
        "mean_abs_n": float(np.mean(np.abs(err3[:, 0]))),
        "mean_abs_e": float(np.mean(np.abs(err3[:, 1]))),
        "mean_abs_d": float(np.mean(np.abs(err3[:, 2]))),
        "std_n": float(np.std(err3[:, 0])),
        "std_e": float(np.std(err3[:, 1])),
        "std_d": float(np.std(err3[:, 2])),
    }


@dataclass
class PlotterESKFJoint:
    # Ground truth and estimates
    rov_gt: TimeSequence[RovNominalCV]
    asv_gt: TimeSequence[AsvNominalState]

    x_upds: TimeSequence[JointEskfState]
    x_preds: TimeSequence[JointEskfState]

    # Measurements (optional)
    z_gnss_asv: TimeSequence[GnssMeasurement] = None
    z_usbl:     TimeSequence[UsblMeasurement] = None
    z_range:    TimeSequence[RangeMeasurement] = None
    z_depth:    TimeSequence[DepthMeasurement] = None

    scenario_name: str = "Joint scenario"
    save_dir: str = None

    # z_preds from run_eskf: {sensor: (z_pred_tseq, z_meas_tseq)}
    z_preds: dict = field(default_factory=dict)

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
        stds = []
        for v in tseq.values:
            P = v.err.cov
            rov_pos_var = np.diag(P)[JointIdx.ROV_POS]
            stds.append(3.0 * np.sqrt(rov_pos_var))
        return np.stack(stds)

    def _asv_est_std(self, tseq: TimeSequence[JointEskfState]) -> np.ndarray:
        stds = []
        for v in tseq.values:
            P = v.err.cov
            asv_pos_var = np.diag(P)[JointIdx.ASV_POS]
            stds.append(3.0 * np.sqrt(asv_pos_var))
        return np.stack(stds)

    # ------------------------
    # Plots
    # ------------------------
    def plot3d(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if self.rov_gt is not None:
            gt_pos = _extract_pos(self.rov_gt, attrgetter("pos"))
            ax.plot(*gt_pos.T, label="ROV ground truth", linestyle="--", color="C1", alpha=0.8)
            ax.scatter(*gt_pos[0], marker="x", color="red", s=60, label="ROV start")

        if self.x_upds is not None and self.x_upds.values:
            est_pos = self._rov_est_pos(self.x_upds)
            ax.plot(*est_pos.T, label="ROV estimate (upd)", color="C0", alpha=0.8)

        if self.asv_gt is not None:
            asv_pos = _extract_pos(self.asv_gt, attrgetter("pos"))
            ax.plot(*asv_pos.T, label="ASV ground truth", color="C2", linestyle="-.", alpha=0.6)
            ax.scatter(*asv_pos[0], marker="^", color="C2", s=60)

        if self.x_upds is not None and self.x_upds.values:
            asv_est_pos = self._asv_est_pos(self.x_upds)
            ax.plot(*asv_est_pos.T, label="ASV estimate (upd)", color="C3", alpha=0.8)

        ax.set_xlabel("North [m]", labelpad=10)
        ax.set_ylabel("East [m]", labelpad=10)
        ax.set_zlabel("Down [m]", labelpad=10)

        ax.invert_zaxis()
        ax.view_init(elev=15, azim=-110)
        ax.set_box_aspect(None)

        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

        ax.grid(True)
        ax.set_title(f"{self.scenario_name} — 3D Trajectories")
        ax.legend(loc='upper right')

        fig.tight_layout()
        return fig

    # ------------------------
    # RMSE (one figure per platform)
    # ------------------------
    def plot_rmse_rov(self):
        if self.x_upds is None or self.rov_gt is None:
            return None

        fig, axs = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=False)

        gt_t = np.asarray(self.rov_gt.times)
        est_t = np.asarray(self.x_upds.times)

        gt_pos = _extract_pos(self.rov_gt, attrgetter("pos"))
        gt_vel = _extract_pos(self.rov_gt, attrgetter("vel"))

        est_pos = self._rov_est_pos(self.x_upds)
        est_vel = self._rov_est_vel(self.x_upds)

        est_pos_i = _interp(est_t, est_pos, gt_t)
        est_vel_i = _interp(est_t, est_vel, gt_t)

        rmse_pos, _ = _error_norm(gt_pos, est_pos_i)
        rmse_vel, _ = _error_norm(gt_vel, est_vel_i)

        axs[0].plot(gt_t, rmse_pos, label="UUV pos RMSE", color="C0")
        axs[0].set_ylabel("RMSE [m]")
        axs[0].set_title("UUV Position RMSE")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(gt_t, rmse_vel, label="UUV vel RMSE", color="C1")
        axs[1].set_ylabel("RMSE [m/s]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_title("UUV Velocity RMSE")
        axs[1].grid(True)
        axs[1].legend()

        fig.suptitle(f"ESKF RMSE — UUV — {self.scenario_name}")
        fig.tight_layout()
        return fig

    def plot_rmse_asv(self):
        if self.x_upds is None or self.asv_gt is None:
            return None

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        gt_t = np.asarray(self.asv_gt.times)
        est_t = np.asarray(self.x_upds.times)

        gt_pos = _extract_pos(self.asv_gt, attrgetter("pos"))
        gt_vel = _extract_pos(self.asv_gt, attrgetter("vel"))

        est_pos = self._asv_est_pos(self.x_upds)
        est_vel = self._asv_est_vel(self.x_upds)

        est_pos_i = _interp(est_t, est_pos, gt_t)
        est_vel_i = _interp(est_t, est_vel, gt_t)

        rmse_pos, _ = _error_norm(gt_pos, est_pos_i)
        rmse_vel, _ = _error_norm(gt_vel, est_vel_i)

        axs[0].plot(gt_t, rmse_pos, label="ASV pos RMSE", color="C3")
        axs[0].set_ylabel("RMSE [m]")
        axs[0].set_title("ASV Position RMSE")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(gt_t, rmse_vel, label="ASV vel RMSE", color="C4")
        axs[1].set_ylabel("RMSE [m/s]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_title("ASV Velocity RMSE")
        axs[1].grid(True)
        axs[1].legend()

        fig.suptitle(f"ESKF RMSE — ASV — {self.scenario_name}")
        fig.tight_layout()
        return fig

    def plot_rov_position(self):
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        labels = ["North [m]", "East [m]", "Down [m]"]

        times_upd = np.array(self.x_upds.times)
        est_pos = self._rov_est_pos(self.x_upds)
        std3 = self._rov_est_std(self.x_upds)

        for i, (ax, lbl) in enumerate(zip(axs, labels)):
            if self.rov_gt is not None:
                gt_times = np.array(self.rov_gt.times)
                gt_pos = _extract_pos(self.rov_gt, attrgetter("pos"))
                ax.plot(gt_times, gt_pos[:, i], label="GT", linestyle="--", color="C1", alpha=0.8)

            ax.plot(times_upd, est_pos[:, i], label="Est", color="C0")
            ax.fill_between(
                times_upd,
                est_pos[:, i] - std3[:, i],
                est_pos[:, i] + std3[:, i],
                alpha=0.2,
                color="C0",
                label="±3σ",
            )

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

        if self.z_gnss_asv is not None:
            gnss_times = np.array(self.z_gnss_asv.times)
            gnss_pos = np.stack([np.asarray(m.pos, dtype=float).reshape(3) for m in self.z_gnss_asv.values])
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
                    label="GNSS meas" if i == 0 else None,
                    zorder=5,
                )

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

    def to_csv_estimated_values(self, filename: str = "estimated_values.csv"):
        if self.save_dir is None or self.x_upds is None or not self.x_upds.values:
            return

        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)

        times = np.asarray(self.x_upds.times, dtype=float)
        rov_pos = self._rov_est_pos(self.x_upds)
        rov_vel = self._rov_est_vel(self.x_upds)
        asv_pos = self._asv_est_pos(self.x_upds)
        asv_vel = self._asv_est_vel(self.x_upds)

        out_file = path / filename
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s",
                "rov_pos_n", "rov_pos_e", "rov_pos_d",
                "rov_vel_n", "rov_vel_e", "rov_vel_d",
                "asv_pos_n", "asv_pos_e", "asv_pos_d",
                "asv_vel_n", "asv_vel_e", "asv_vel_d",
            ])

            for i in range(len(times)):
                writer.writerow([
                    times[i],
                    rov_pos[i, 0], rov_pos[i, 1], rov_pos[i, 2],
                    rov_vel[i, 0], rov_vel[i, 1], rov_vel[i, 2],
                    asv_pos[i, 0], asv_pos[i, 1], asv_pos[i, 2],
                    asv_vel[i, 0], asv_vel[i, 1], asv_vel[i, 2],
                ])

    def collect_statistics_rows(self) -> list[dict]:
        if self.x_upds is None or not self.x_upds.values:
            return []

        rows = []
        times = np.asarray(self.x_upds.times, dtype=float)
        #nis_stats = self._compute_nis_stats()

        # ---------- ROV ----------
        if self.rov_gt is not None:
            gt_pos = np.stack([self.rov_gt.at_time(t).pos for t in times])
            gt_vel = np.stack([self.rov_gt.at_time(t).vel for t in times])
            est_pos = self._rov_est_pos(self.x_upds)
            est_vel = self._rov_est_vel(self.x_upds)

            pos_err = est_pos - gt_pos
            vel_err = est_vel - gt_vel

            pos_stats = _error_stats_from_errvec(pos_err)
            vel_stats = _error_stats_from_errvec(vel_err)

            pl_gt = _path_length(gt_pos)
            pl_est = _path_length(est_pos)

            nees_pos_vals, nees_vel_vals = [], []
            for t, x in self.x_upds.items():
                gt = JointNominalState(asv=self.asv_gt.at_time(t), rov=self.rov_gt.at_time(t))
                err_gauss = x.get_err_gauss(gt)
                err = np.asarray(err_gauss.mean)
                P = err_gauss.cov

                try:
                    e = err[JointIdx.ROV_POS]
                    Pp = P[JointIdx.ROV_POS, JointIdx.ROV_POS]
                    nees_pos_vals.append(float(e @ np.linalg.solve(Pp, e)))
                except np.linalg.LinAlgError:
                    pass

                try:
                    e = err[JointIdx.ROV_VEL]
                    Pp = P[JointIdx.ROV_VEL, JointIdx.ROV_VEL]
                    nees_vel_vals.append(float(e @ np.linalg.solve(Pp, e)))
                except np.linalg.LinAlgError:
                    pass

            rows.append({
                **{f"pos_{k}": v for k, v in pos_stats.items()},
                # "path_length_gt": float(pl_gt),
                # "path_length_est": float(pl_est),
                #"path_length_error_pct": float(100.0 * abs(pl_est - pl_gt) / pl_gt) if pl_gt > 0 else np.nan,
                "mean_tanees_pos": float(np.nanmean(nees_pos_vals)) if nees_pos_vals else np.nan,
                "mean_tanees_vel": float(np.nanmean(nees_vel_vals)) if nees_vel_vals else np.nan,
                "tanees_pos": float(np.nanmean(nees_pos_vals)) if nees_pos_vals else np.nan,
                "tanees_vel": float(np.nanmean(nees_vel_vals)) if nees_vel_vals else np.nan,
                "scenario": self.scenario_name,
                "platform": "ROV",
                #**{f"vel_{k}": v for k, v in vel_stats.items()},
                #**nis_stats,
            })

        # ---------- ASV ----------
        if self.asv_gt is not None:
            gt_pos = np.stack([self.asv_gt.at_time(t).pos for t in times])
            gt_vel = np.stack([self.asv_gt.at_time(t).vel for t in times])
            est_pos = self._asv_est_pos(self.x_upds)
            est_vel = self._asv_est_vel(self.x_upds)

            pos_err = est_pos - gt_pos
            vel_err = est_vel - gt_vel

            pos_stats = _error_stats_from_errvec(pos_err)
            vel_stats = _error_stats_from_errvec(vel_err)

            pl_gt = _path_length(gt_pos)
            pl_est = _path_length(est_pos)

            nees_pos_vals, nees_vel_vals = [], []
            for t, x in self.x_upds.items():
                gt = JointNominalState(asv=self.asv_gt.at_time(t), rov=self.rov_gt.at_time(t))
                err_gauss = x.get_err_gauss(gt)
                err = np.asarray(err_gauss.mean)
                P = err_gauss.cov

                try:
                    e = err[JointIdx.ASV_POS]
                    Pp = P[JointIdx.ASV_POS, JointIdx.ASV_POS]
                    nees_pos_vals.append(float(e @ np.linalg.solve(Pp, e)))
                except np.linalg.LinAlgError:
                    pass

                try:
                    e = err[JointIdx.ASV_VEL]
                    Pp = P[JointIdx.ASV_VEL, JointIdx.ASV_VEL]
                    nees_vel_vals.append(float(e @ np.linalg.solve(Pp, e)))
                except np.linalg.LinAlgError:
                    pass

            rows.append({
                **{f"pos_{k}": v for k, v in pos_stats.items()},
                # "path_length_gt": float(pl_gt),
                # "path_length_est": float(pl_est),
                # "path_length_error_pct": float(100.0 * abs(pl_est - pl_gt) / pl_gt) if pl_gt > 0 else np.nan,
                "mean_tanees_pos": float(np.nanmean(nees_pos_vals)) if nees_pos_vals else np.nan,
                "mean_tanees_vel": float(np.nanmean(nees_vel_vals)) if nees_vel_vals else np.nan,
                "tanees_pos": float(np.nanmean(nees_pos_vals)) if nees_pos_vals else np.nan,
                "tanees_vel": float(np.nanmean(nees_vel_vals)) if nees_vel_vals else np.nan,
                "scenario": self.scenario_name,
                "platform": "ASV",
                # **{f"vel_{k}": v for k, v in vel_stats.items()},
                #**nis_stats,
            })

        return rows

    def export_statistics(self, filename: str = "eskf_statistics.csv"):
        """
        Export per-platform statistics (position + velocity) to CSV in save_dir.
        """
        if self.save_dir is None:
            return None

        rows = self.collect_statistics_rows()
        if not rows:
            return None

        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)

        out_file = path / filename
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return out_file

    def _compute_nis_stats(self) -> dict:
        if not self.z_preds:
            return {}

        sensor_cfg = {
            "gnss":  (3, False),
            "usbl":  (2, True),
            "range": (1, False),
            "depth": (1, False),
        }

        stats = {}
        for key, (dof, wrap_az) in sensor_cfg.items():
            if key not in self.z_preds:
                continue

            z_pred_tseq, z_meas_tseq = self.z_preds[key]
            nis_vals = []
            for t, z_pred in z_pred_tseq.items():
                if t not in z_meas_tseq:
                    continue
                z_p = np.asarray(z_pred.mean).reshape(-1)
                z_m = np.asarray(z_meas_tseq.get_t(t)).reshape(-1)
                innov = z_m - z_p
                if wrap_az:
                    innov[0] = wrap_to_pi(innov[0])
                nis = float(innov @ np.linalg.solve(z_pred.cov, innov))
                nis_vals.append(nis)

            stats[f"anis_{key}"] = float(np.nanmean(nis_vals)) if nis_vals else np.nan
            stats[f"nis_samples_{key}"] = int(len(nis_vals))
            stats[f"nis_dof_{key}"] = int(dof)

        return stats

    @staticmethod
    def _chi2_bands(ax, times, dof, alpha=0.95):
        ci_lo, ci_hi = chi2.interval(alpha, dof)
        ci_med = chi2.ppf(0.5, dof)
        ax.axhline(ci_lo,  color="tab:orange", ls="--", alpha=0.7,
                   label=f"χ²({dof}) {alpha:.0%} CI")
        ax.axhline(ci_med, color="tab:green",  ls="--", alpha=0.7,
                   label=f"χ²({dof}) median")
        ax.axhline(ci_hi,  color="tab:orange", ls="--", alpha=0.7)

    def plot_nees(self):
        if self.rov_gt is None or self.asv_gt is None or self.x_upds is None:
            return None

        times = []
        nees_rov_pos, nees_rov_vel = [], []
        nees_asv_pos, nees_asv_vel = [], []

        for t, x in self.x_upds.items():
            gt = JointNominalState(
                asv=self.asv_gt.at_time(t),
                rov=self.rov_gt.at_time(t),
            )
            err_gauss = x.get_err_gauss(gt)
            err = np.asarray(err_gauss.mean)
            P = err_gauss.cov

            rov_pos_err = err[JointIdx.ROV_POS]
            rov_pos_P = P[JointIdx.ROV_POS, JointIdx.ROV_POS]
            nees_rov_pos.append(float(rov_pos_err @ np.linalg.solve(rov_pos_P, rov_pos_err)))

            rov_vel_err = err[JointIdx.ROV_VEL]
            rov_vel_P = P[JointIdx.ROV_VEL, JointIdx.ROV_VEL]
            nees_rov_vel.append(float(rov_vel_err @ np.linalg.solve(rov_vel_P, rov_vel_err)))

            asv_pos_err = err[JointIdx.ASV_POS]
            asv_pos_P = P[JointIdx.ASV_POS, JointIdx.ASV_POS]
            nees_asv_pos.append(float(asv_pos_err @ np.linalg.solve(asv_pos_P, asv_pos_err)))

            asv_vel_err = err[JointIdx.ASV_VEL]
            asv_vel_P = P[JointIdx.ASV_VEL, JointIdx.ASV_VEL]
            nees_asv_vel.append(float(asv_vel_err @ np.linalg.solve(asv_vel_P, asv_vel_err)))

            times.append(t)

        times = np.array(times)
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 9))

        axs[0].semilogy(times, nees_rov_pos, color="C0", label="NEES")
        self._chi2_bands(axs[0], times, dof=3)
        axs[0].set_ylabel("NEES ROV pos (3 DOF)")
        axs[0].legend()

        axs[1].semilogy(times, nees_rov_vel, color="C1", label="NEES")
        self._chi2_bands(axs[1], times, dof=3)
        axs[1].set_ylabel("NEES ROV vel (3 DOF)")
        axs[1].legend()

        axs[2].semilogy(times, nees_asv_pos, color="C3", label="NEES")
        self._chi2_bands(axs[2], times, dof=3)
        axs[2].set_ylabel("NEES ASV pos (3 DOF)")
        axs[2].legend()

        axs[3].semilogy(times, nees_asv_vel, color="C4", label="NEES")
        self._chi2_bands(axs[3], times, dof=3)
        axs[3].set_ylabel("NEES ASV vel (3 DOF)")
        axs[3].set_xlabel("Time [s]")
        axs[3].legend()

        axs[0].set_title(f"{self.scenario_name} — NEES (pos + vel)")
        fig.tight_layout()
        return fig

    def plot_nis(self):
        if not self.z_preds:
            return None

        sensor_cfg = {
            "gnss":  ("GNSS (3 DOF)",  3,  "C0", False),
            "usbl":  ("USBL (2 DOF)",  2,  "C4", True),
            "range": ("Range (1 DOF)", 1,  "C2", False),
            "depth": ("Depth (1 DOF)", 1,  "C5", False),
        }
        active = [(k, *sensor_cfg[k]) for k in sensor_cfg if k in self.z_preds]
        if not active:
            return None

        fig, axs = plt.subplots(len(active), 1, sharex=True,
                                figsize=(10, 2.5 * len(active)))
        if len(active) == 1:
            axs = [axs]

        for ax, (key, label, dof, color, wrap_az) in zip(axs, active):
            z_pred_tseq, z_meas_tseq = self.z_preds[key]
            times, nis_vals = [], []
            for t, z_pred in z_pred_tseq.items():
                if t not in z_meas_tseq:
                    continue
                z_p = np.asarray(z_pred.mean).reshape(-1)
                z_m = np.asarray(z_meas_tseq.get_t(t)).reshape(-1)
                innov = z_m - z_p
                if wrap_az:
                    innov[0] = wrap_to_pi(innov[0])
                nis = float(innov @ np.linalg.solve(z_pred.cov, innov))
                times.append(t)
                nis_vals.append(nis)

            times = np.array(times)
            ax.semilogy(times, nis_vals, ".", color=color, markersize=4,
                        label="NIS")
            self._chi2_bands(ax, times, dof=dof)
            ax.set_ylabel(label)
            ax.legend()

        axs[0].set_title(f"{self.scenario_name} — NIS")
        axs[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        return fig

    def show(self):
        self.to_csv_estimated_values()
        self.export_statistics()
        self._save(self.plot3d(), "3d_trajectory")
        self._save(self.plot_rmse_rov(), "rmse_rov")
        self._save(self.plot_rmse_asv(), "rmse_asv")
        # self._save(self.plot_rov_position(), "rov_position")
        # self._save(self.plot_asv_position(), "asv_position")
        # self._save(self.plot_rov_position_error(), "rov_position_error")
        self._save(self.plot_nees(), "nees")
        # self._save(self.plot_nis(), "nis")
        # self._save(self.plot_usbl_measurements(), "usbl")
        # self._save(self.plot_range_measurements(), "range")
        plt.show(block=False)

    def _save(self, fig, name: str):
        if fig is None:
            return
        if self.save_dir is None:
            return
        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f"{name}.png", dpi=150, bbox_inches="tight")
