#!/usr/bin/env python3
"""
Generate ground-truth plots for the figure-8 trajectory scenario.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from exp1_plots import TrajectoryType  # reuse the same TrajectoryType enum

# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.serif": ["CMU Serif", "Computer Modern", "cmr10", "DejaVu Serif"],
#         "mathtext.fontset": "cm",
#     }
# )
plt.rcParams.update(    {        "font.family": "serif",        "font.serif": ["CMU Serif", "DejaVu Serif"],        "mathtext.fontset": "cm",        "axes.formatter.use_mathtext": True,        "axes.unicode_minus": False,    })
# Ensure workspace src/ is on sys.path when running this file directly.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = WORKSPACE_ROOT / "src"
if str(WORKSPACE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_SRC))

from tracking_and_navigation.generate_trajectories import (
    generate_trajectories,
    TrajectoryType,
)


def _yaw_from_rotmat(R_nb: np.ndarray) -> float:
    return float(math.atan2(R_nb[1, 0], R_nb[0, 0]))


def _extract_positions(tseq) -> np.ndarray:
    return np.stack([s.pos for s in tseq.values])


def _extract_times(tseq) -> np.ndarray:
    return np.asarray(tseq.times, dtype=float)


def wrap_and_break(angle_rad):
    angle_deg = np.degrees(angle_rad)

    # wrap to [-180, 180]
    angle_deg = (angle_deg + 180) % 360 - 180

    # detect jumps
    jumps = np.abs(np.diff(angle_deg)) > 180

    angle_deg = angle_deg.copy()
    angle_deg[1:][jumps] = np.nan  # break the line

    return angle_deg

def wrap_360_break(angle_rad):
    angle_deg = np.degrees(angle_rad)
    angle_deg = angle_deg % 360

    jumps = np.abs(np.diff(angle_deg)) > 180
    angle_deg = angle_deg.copy()
    angle_deg[1:][jumps] = np.nan  # break lines

    return angle_deg


def deg_formatter(x, pos):
    return f"{int(x)}°"



def _plot_trajectories(asv_pos: np.ndarray, rov_pos: np.ndarray, t: np.ndarray,
                       out_path: Path | None, title_traj: str):

    fig, axs = plt.subplots(2, 1, figsize=(7.5, 7.5),
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axs[0]

    ax.plot(asv_pos[:, 0], asv_pos[:, 1], color="C0", label="USV")
    ax.plot(rov_pos[:, 0], rov_pos[:, 1], color="C1", label="UUV")

    ax.scatter(asv_pos[0, 0], asv_pos[0, 1], color="C0", marker="o", s=60, label="USV start")
    ax.scatter(rov_pos[0, 0], rov_pos[0, 1], color="C1", marker="o", s=60, label="UUV start")

    def add_path_arrows(pos, color):
        n = len(pos)
        idxs = np.linspace(0, n - 2, 6, dtype=int)
        for i in idxs:
            ax.annotate(
                "",
                xy=pos[i + 1, :2],
                xytext=pos[i, :2],
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.7, lw=1.5),
            )

    add_path_arrows(asv_pos, "C0")
    add_path_arrows(rov_pos, "C1")

    key_times = [5, 10, 30, 60, 120]
    for kt in key_times:
        idx = np.argmin(np.abs(t - kt))
        ax.annotate(
            "",
            xy=rov_pos[idx, :2],
            xytext=asv_pos[idx, :2],
            arrowprops=dict(arrowstyle="->", color="k", linestyle="--", alpha=0.6, lw=1.2),
        )

    ax.set_xlabel("North [m]")
    ax.set_ylabel("East [m]")
    ax.set_title(f"Ground-truth top-down view")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")

    # ---- Depth vs time ----
    ax2 = axs[1]
    ax2.plot(t, asv_pos[:, 2], color="C0", label="USV")
    ax2.plot(t, rov_pos[:, 2], color="C1", label="UUV")
    ax2.invert_yaxis()
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Depth [m]")
    ax2.set_title(f"Ground-truth depth profiles")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize="small")

    fig.suptitle(f"{title_traj} trajectories", fontsize=14)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")

    return fig


# -----------------------------
# UPDATED: geometry plots
# -----------------------------
def _plot_geometry(t: np.ndarray, bearing: np.ndarray, heading: np.ndarray, slant_range: np.ndarray,
                   out_path: Path | None, title_traj: str):

    fig, axs = plt.subplots(3, 1, figsize=(7.5, 7.5), sharex=True)
    # bearing = np.unwrap(bearing)
    # heading = np.unwrap(heading)
    # bearing_plot = wrap_and_break(bearing)
    # heading_plot = wrap_and_break(heading)
    bearing_plot = wrap_360_break(bearing)
    heading_plot = wrap_360_break(heading)

    axs[0].plot(t, bearing_plot, color="C2")
    axs[1].plot(t, heading_plot, color="C0")


    # ---- Bearing ----
    # axs[0].plot(t, np.degrees(bearing), color="C2")
    axs[0].set_ylabel("Azimuth [deg]")
    axs[0].set_title("Azimuth angle (USV to UUV)")
    axs[0].grid(True, alpha=0.3)
    # axs[0].set_ylim(-180, 180)
    # axs[0].set_yticks(["-180°", "-90°", "0°", "90°", "180°"])

    # ---- Heading ----
    # axs[1].plot(t, np.degrees(heading), color="C0")
    axs[1].set_ylabel("Heading [deg]")
    axs[1].set_title("USV heading")
    axs[1].grid(True, alpha=0.3)
    # axs[1].set_ylim(-180, 180)
    # axs[1].set_yticks(["-180°", "-90°", "0°", "90°", "180°"])
    ticks = [0, 90, 180, 270, 360]
    for ax in axs[:2]:    
        # ax.set_ylim(-180, 180)    
        ax.set_ylim(0, 360)
        ax.set_yticks(ticks)    
        ax.yaxis.set_major_formatter(FuncFormatter(deg_formatter))

    # ---- Slant range ----
    axs[2].plot(t, slant_range, color="C1")
    axs[2].set_ylabel("Slant range [m]")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_title("USV-UUV slant range")
    axs[2].grid(True, alpha=0.3)


    fig.suptitle(f"Measurement geometry of {title_traj}", fontsize=14)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate ground-truth plots.")
    parser.add_argument("--trajectory", type=str, default="figure_8",
                        choices=[tt.value for tt in TrajectoryType],
                        help="Type of trajectory to generate (default: figure_8)")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=0.01)
    # parser.add_argument("--out-dir", type=str, default="plots//gt_300s_figure_8")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.trajectory == "figure_8":
        traj_type = TrajectoryType.FIGURE_8
    elif args.trajectory == "linear_turns":
        traj_type = TrajectoryType.LINEAR_TURNS
    else:
        raise ValueError(f"Invalid trajectory type: {args.trajectory}")
    asv_tseq, rov_tseq, _ = generate_trajectories(
        duration=args.duration,
        dt=args.dt,
        trajectory_type=traj_type,
    )
    out_dir = Path(f"plots/{traj_type.value}/gt_{int(args.duration)}s_360deg")

    t = _extract_times(asv_tseq)
    asv_pos = _extract_positions(asv_tseq)
    rov_pos = _extract_positions(rov_tseq)

    d = rov_pos - asv_pos
    bearing = np.array([math.atan2(de, dn) for dn, de, _ in d])

    heading = np.array([_yaw_from_rotmat(s.ori.as_rotmat()) for s in asv_tseq.values])

    slant_range = np.linalg.norm(d, axis=1)

    # out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "sim_trajectories.svg"
    geom_path = out_dir / "sim_geometry.svg"

    traj_title_map = {
        TrajectoryType.FIGURE_8: "Circular/Figure-8",
        TrajectoryType.LINEAR_TURNS: "Linear/Linear-turns",
    }
    title_traj = traj_title_map[traj_type]
    _plot_trajectories(asv_pos, rov_pos, t, traj_path, title_traj)
    _plot_geometry(t, bearing, heading, slant_range, geom_path, title_traj)

    print(f"[OK] Wrote {traj_path}")
    print(f"[OK] Wrote {geom_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
