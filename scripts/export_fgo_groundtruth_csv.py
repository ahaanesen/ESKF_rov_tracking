#!/usr/bin/env python3
"""
Export ASV/ROV ground truth trajectories to CSV for FGO error analysis.

Usage example:
    PYTHONPATH=src:$PYTHONPATH python3 scripts/export_fgo_groundtruth_csv.py \
  --out-dir /tmp/fgo_dataset_8_var_speed_gt \
  --duration 300 \
  --dt 0.1 \
  --seed 42 \
  --rov-id 1 \
  --epoch-sec 1700000000

Outputs:
  - <out_dir>/asv_ground_truth.csv
  - <out_dir>/rov_ground_truth.csv
  - <out_dir>/gt_metadata.json
"""

import argparse
import csv
import json
import os
import math
import numpy as np

from tracking_and_navigation.generate_trajectories import generate_trajectories


def sim_time_to_ros_stamp(sim_t: float, epoch_sec: int) -> tuple[int, int]:
    """Match rosbag exporter stamp conversion exactly."""
    t = epoch_sec + sim_t
    sec = int(t)
    nsec = int((t - sec) * 1e9)
    return sec, nsec


def to_ns(sim_t: float, epoch_sec: int) -> int:
    """Match rosbag exporter nanosecond conversion exactly."""
    return int((epoch_sec + sim_t) * 1e9)


def yaw_from_rotmat(R_nb: np.ndarray) -> float:
    return float(math.atan2(R_nb[1, 0], R_nb[0, 0]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rov-id", type=int, default=1)
    p.add_argument("--epoch-sec", type=int, default=1700000000)
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    asv_tseq, rov_tseq, _ = generate_trajectories(duration=args.duration, dt=args.dt)

    asv_csv = os.path.join(args.out_dir, "asv_ground_truth.csv")
    rov_csv = os.path.join(args.out_dir, "rov_ground_truth.csv")
    meta_json = os.path.join(args.out_dir, "gt_metadata.json")

    with open(asv_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_sim_sec", "t_ros_sec", "t_ros_ns",
            "x_n", "y_e", "z_d",
            "vx_n", "vy_e", "vz_d",
            "yaw_rad"
        ])
        for t, s in asv_tseq.items():
            R_nb = s.ori.as_rotmat()
            yaw = yaw_from_rotmat(R_nb)
            t_ros_sec, t_ros_nsec = sim_time_to_ros_stamp(float(t), args.epoch_sec)
            t_ns = to_ns(float(t), args.epoch_sec)
            w.writerow([
                float(t), t_ros_sec + t_ros_nsec * 1e-9, t_ns,
                float(s.pos[0]), float(s.pos[1]), float(s.pos[2]),
                float(s.vel[0]), float(s.vel[1]), float(s.vel[2]),
                yaw
            ])

    with open(rov_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_sim_sec", "t_ros_sec", "t_ros_ns", "rov_id",
            "x_n", "y_e", "z_d",
            "vx_n", "vy_e", "vz_d"
        ])
        for t, s in rov_tseq.items():
            t_ros_sec, t_ros_nsec = sim_time_to_ros_stamp(float(t), args.epoch_sec)
            t_ns = to_ns(float(t), args.epoch_sec)
            w.writerow([
                float(t), t_ros_sec + t_ros_nsec * 1e-9, t_ns, int(args.rov_id),
                float(s.pos[0]), float(s.pos[1]), float(s.pos[2]),
                float(s.vel[0]), float(s.vel[1]), float(s.vel[2]),
            ])

    with open(meta_json, "w") as f:
        json.dump({
            "duration_sec": args.duration,
            "dt_sec": args.dt,
            "seed": args.seed,
            "rov_id": args.rov_id,
            "epoch_sec": args.epoch_sec,
            "frame": "NED",
            "timestamp_alignment": {
                "t_ros_sec_definition": "epoch_sec + t_sim_sec",
                "t_ros_ns_definition": "int((epoch_sec + t_sim_sec) * 1e9)",
                "stamp_split": "sec=int(t), nsec=int((t-sec)*1e9)",
            },
            "notes": "For exact alignment, use same duration/dt/seed/epoch_sec as rosbag exporter."
        }, f, indent=2)

    print(f"[OK] Wrote {asv_csv}")
    print(f"[OK] Wrote {rov_csv}")
    print(f"[OK] Wrote {meta_json}")

if __name__ == "__main__":
    main()