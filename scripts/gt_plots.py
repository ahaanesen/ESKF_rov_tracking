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


# -----------------------------
# UPDATED: trajectory plot
# -----------------------------
def _plot_trajectories(asv_pos: np.ndarray, rov_pos: np.ndarray, t: np.ndarray,
                       out_path: Path | None):

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    ax.plot(asv_pos[:, 0], asv_pos[:, 1], color="C0", label="USV (figure-8)")
    ax.plot(rov_pos[:, 0], rov_pos[:, 1], color="C1", label="ROV (circular sweep)")

    # Start markers only
    ax.scatter(asv_pos[0, 0], asv_pos[0, 1], color="C0", marker="o", s=60, label="USV start")
    ax.scatter(rov_pos[0, 0], rov_pos[0, 1], color="C1", marker="o", s=60, label="ROV start")

    # Direction arrows along trajectories
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

    # USV -> ROV arrows at key timestamps
    key_times = [5, 10, 30, 60, 120]

    for kt in key_times:
        idx = np.argmin(np.abs(t - kt))
        p_usv = asv_pos[idx]
        p_rov = rov_pos[idx]

        ax.annotate(
            "",
            xy=p_rov[:2],
            xytext=p_usv[:2],
            arrowprops=dict(
                arrowstyle="->",
                color="k",
                linestyle="--",
                alpha=0.6,
                lw=1.2,
            ),
        )

    ax.set_xlabel("North [m]")
    ax.set_ylabel("East [m]")
    ax.set_title("Ground-truth trajectories (top-down NE view)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")

    return fig


# -----------------------------
# UPDATED: geometry plots
# -----------------------------
def _plot_geometry(t: np.ndarray, bearing: np.ndarray, heading: np.ndarray, slant_range: np.ndarray,
                   out_path: Path | None):

    fig, axs = plt.subplots(3, 1, figsize=(8.0, 7.5), sharex=True)
    bearing = np.unwrap(bearing)
    heading = np.unwrap(heading)

    # ---- Bearing ----
    axs[0].plot(t, np.degrees(bearing), color="C2")
    axs[0].set_ylabel("Bearing [deg]")
    axs[0].set_title("Bearing angle (USV to ROV)")
    axs[0].grid(True, alpha=0.3)

    # ---- Heading ----
    axs[1].plot(t, np.degrees(heading), color="C0")
    axs[1].set_ylabel("Heading [deg]")
    axs[1].set_title("USV heading")
    axs[1].grid(True, alpha=0.3)

    # ---- Slant range ----
    axs[2].plot(t, slant_range, color="C1")
    axs[2].set_ylabel("Slant range [m]")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_title("USV-ROV slant range")
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate figure-8 ground-truth plots.")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--out-dir", type=str, default="plots/gt_300s_fig8_unwrapped")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    asv_tseq, rov_tseq, _ = generate_trajectories(
        duration=args.duration,
        dt=args.dt,
        trajectory_type=TrajectoryType.FIGURE_8,
    )

    t = _extract_times(asv_tseq)
    asv_pos = _extract_positions(asv_tseq)
    rov_pos = _extract_positions(rov_tseq)

    d = rov_pos - asv_pos
    bearing = np.array([math.atan2(de, dn) for dn, de, _ in d])

    heading = np.array([_yaw_from_rotmat(s.ori.as_rotmat()) for s in asv_tseq.values])

    slant_range = np.linalg.norm(d, axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "sim_trajectories.svg"
    geom_path = out_dir / "sim_geometry.svg"

    _plot_trajectories(asv_pos, rov_pos, t, traj_path)
    _plot_geometry(t, bearing, heading, slant_range, geom_path)

    print(f"[OK] Wrote {traj_path}")
    print(f"[OK] Wrote {geom_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
