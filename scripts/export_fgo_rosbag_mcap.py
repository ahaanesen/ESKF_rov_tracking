#!/usr/bin/env python3
"""
Export synthetic ASV/ROV data from ESKF generators to ROS 2 bag (MCAP)
compatible with microamp_fgo_rov_tracking.

Writes:
  - sensor_msgs/msg/Imu                      -> microampere/imu/data
  - blueboat_interfaces/msg/GNSSNavPvt       -> microampere/gnss/nav_pvt
  - blueboat_interfaces/msg/USBL             -> microampere/sensors/usbl
  - blueboat_interfaces/msg/AcousticCommReceive (optional)

Usage example:
PYTHONPATH=src:$PYTHONPATH python3 scripts/export_fgo_rosbag_mcap.py \
    --out /tmp/fgo_dataset \
    --duration 300 \
    --seed 42 \
    --datum-lat 60.3913 \
    --datum-lon 5.3221 \
    --datum-h 0.0 \
    --write-acoustic-rx false
"""

import argparse
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Point

# ESKF repo imports
from tracking_and_navigation.generate_trajectories import generate_trajectories
from tracking_and_navigation.generate_measurements import MeasurementGenerator

# ----------------------------
# WGS84 helpers (NED <-> LLA)
# ----------------------------

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

def lla_to_ecef(lat_rad: float, lon_rad: float, h_m: float) -> np.ndarray:
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + h_m) * cos_lat * cos_lon
    y = (N + h_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=float)

def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # Iterative solution (stable for our use)
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    h = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - N
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * N / (N + h)))
    return lat, lon, h

def ned_to_ecef_delta(n: float, e: float, d: float, lat0_rad: float, lon0_rad: float) -> np.ndarray:
    sL = math.sin(lat0_rad)
    cL = math.cos(lat0_rad)
    sO = math.sin(lon0_rad)
    cO = math.cos(lon0_rad)

    # NED->ECEF rotation is transpose(ECEF->NED)
    R = np.array([
        [-sL * cO, -sO, -cL * cO],
        [-sL * sO,  cO, -cL * sO],
        [ cL,       0.0, -sL    ],
    ], dtype=float)
    # ECEF delta = R^T * ned
    ned = np.array([n, e, d], dtype=float)
    return R.T @ ned

def ned_to_lla(n: float, e: float, d: float, lat0_deg: float, lon0_deg: float, h0_m: float) -> Tuple[float, float, float]:
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    ecef0 = lla_to_ecef(lat0, lon0, h0_m)
    de = ned_to_ecef_delta(n, e, d, lat0, lon0)
    ecef = ecef0 + de
    lat, lon, h = ecef_to_lla(ecef[0], ecef[1], ecef[2])
    return math.degrees(lat), math.degrees(lon), h

# ----------------------------
# ROS time helpers
# ----------------------------

def sim_time_to_ros_time(sim_t: float, epoch_sec: int) -> TimeMsg:
    t = epoch_sec + sim_t
    sec = int(t)
    nsec = int((t - sec) * 1e9)
    msg = TimeMsg()
    msg.sec = sec
    msg.nanosec = nsec
    return msg

def to_ns(sim_t: float, epoch_sec: int) -> int:
    return int((epoch_sec + sim_t) * 1e9)

# ----------------------------
# Data utilities / senfuslib compatibility
# ----------------------------

# Utility to convert sequence of (t, z) to pairs for nearest-time lookup
def as_pairs(ts):
    # adapt to senfuslib TimeSequence
    if hasattr(ts, "items"):      # common pattern
        return list(ts.items())
    if hasattr(ts, "_items"):     # fallback
        return list(ts._items)
    if hasattr(ts, "data"):       # fallback
        return list(ts.data)
    # last resort: maybe it is iterable in your version
    return list(ts)

# Utility to convert measurement to array, works for senfuslib NamedArray subclasses
def meas_as_array(m):
    # Works for senfuslib NamedArray subclasses
    return np.asarray(m, dtype=float).reshape(-1)

# ----------------------------
# Event model
# ----------------------------

@dataclass
class Event:
    t: float
    kind: str
    payload: object

def str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y", "on")

# ----------------------------
# Main export
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output rosbag directory")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=0.1, help="Trajectory dt")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--datum-lat", type=float, default=60.3913)
    parser.add_argument("--datum-lon", type=float, default=5.3221)
    parser.add_argument("--datum-h", type=float, default=0.0)

    parser.add_argument("--sound-speed", type=float, default=1500.0)
    parser.add_argument("--rov-id", type=int, default=1)

    parser.add_argument("--imu-rate", type=float, default=100.0)
    parser.add_argument("--gnss-rate", type=float, default=1.0)
    parser.add_argument("--usbl-rate", type=float, default=1.0)
    parser.add_argument("--depth-rate", type=float, default=5.0)
    parser.add_argument("--range-rate", type=float, default=1.0)

    parser.add_argument("--imu-acc-std", type=float, default=0.02)
    parser.add_argument("--imu-gyro-std", type=float, default=0.002)
    parser.add_argument("--gnss-std-ne", type=float, default=0.8)
    parser.add_argument("--gnss-std-d", type=float, default=1.2)
    parser.add_argument("--usbl-std-rad", type=float, default=0.03)
    parser.add_argument("--range-std-m", type=float, default=0.5)
    parser.add_argument("--depth-std-m", type=float, default=0.2)

    parser.add_argument("--h-acc-mm", type=int, default=800)
    parser.add_argument("--v-acc-mm", type=int, default=1200)

    parser.add_argument("--write-acoustic-rx", type=str, default="false")
    parser.add_argument("--epoch-sec", type=int, default=1700000000)

    # Topics (your configured defaults)
    parser.add_argument("--topic-imu", type=str, default="microampere/imu/data")
    parser.add_argument("--topic-gnss", type=str, default="microampere/gnss/nav_pvt")
    parser.add_argument("--topic-usbl", type=str, default="microampere/sensors/usbl")
    parser.add_argument("--topic-acoustic", type=str, default="microampere/acoustic/receive")

    args = parser.parse_args()
    np.random.seed(args.seed)
    write_acoustic = str2bool(args.write_acoustic_rx)

    # Dynamic message classes (avoids hard dependency at import time)
    ImuMsg = get_message("sensor_msgs/msg/Imu")
    GNSSMsg = get_message("blueboat_interfaces/msg/GNSSNavPvt")
    USBLMsg = get_message("blueboat_interfaces/msg/USBL")
    AcousticMsg = get_message("blueboat_interfaces/msg/AcousticCommReceive")

    # 1) Generate trajectories and measurements from ESKF tooling
    asv_gt, rov_gt, _imu_gt = generate_trajectories(duration=args.duration, dt=args.dt)
    mg = MeasurementGenerator(asv_gt, rov_gt)

    lever_arm = np.array([0.0, 0.0, 1.5], dtype=float)  # match env.usbl_offset_z default

    imu_seq = mg.generate_imu_asv(
        accm_std=args.imu_acc_std,
        gyro_std=args.imu_gyro_std,
        rate_hz=args.imu_rate,
    )
    gnss_seq = mg.generate_gnss_asv(
        std_ne=args.gnss_std_ne,
        std_d=args.gnss_std_d,
        lever_arm=np.zeros(3),  # keep simple; can set lever arm if desired
        rate_hz=args.gnss_rate,
    )
    usbl_seq = mg.generate_usbl(
        std_rad=args.usbl_std_rad,
        lever_arm=lever_arm,
        rate_hz=args.usbl_rate,
    )
    range_seq = mg.generate_range(
        std_m=args.range_std_m,
        lever_arm=lever_arm,
        rate_hz=args.range_rate,
    )
    depth_seq = mg.generate_depth(
        std_m=args.depth_std_m,
        rate_hz=args.depth_rate,
    )

    # Build nearest-time lookup for depth/range at USBL times
    imu_pairs = as_pairs(imu_seq)
    gnss_pairs = as_pairs(gnss_seq)
    usbl_pairs = as_pairs(usbl_seq)
    range_pairs = as_pairs(range_seq)
    depth_pairs = as_pairs(depth_seq)

    range_times = np.array([t for t, _ in range_pairs], dtype=float)
    range_vals = [z for _, z in range_pairs]
    depth_times = np.array([t for t, _ in depth_pairs], dtype=float)
    depth_vals = [z for _, z in depth_pairs]

    def nearest(times: np.ndarray, vals: List, t: float):
        idx = int(np.argmin(np.abs(times - t)))
        return vals[idx]

    events: List[Event] = []

    for t, z in imu_pairs:
        events.append(Event(float(t), "imu", z))

    for t, z in gnss_pairs:
        events.append(Event(float(t), "gnss", z))

    for t, z in usbl_pairs:
        events.append(Event(float(t), "usbl", z))

    events.sort(key=lambda e: e.t)

    # 2) Open rosbag2 writer with MCAP storage
    storage_options = rosbag2_py.StorageOptions(
        uri=args.out,
        storage_id="mcap",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=args.topic_imu,
            type="sensor_msgs/msg/Imu",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=args.topic_gnss,
            type="blueboat_interfaces/msg/GNSSNavPvt",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=args.topic_usbl,
            type="blueboat_interfaces/msg/USBL",
            serialization_format="cdr",
        )
    )
    if write_acoustic:
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=args.topic_acoustic,
                type="blueboat_interfaces/msg/AcousticCommReceive",
                serialization_format="cdr",
            )
        )

    # 3) Serialize and write
    for e in events:
        stamp = sim_time_to_ros_time(e.t, args.epoch_sec)
        stamp_ns = to_ns(e.t, args.epoch_sec)

        if e.kind == "imu":
            m = ImuMsg()
            m.header = Header()
            m.header.stamp = stamp
            m.header.frame_id = "base_link"

            # ESKF ImuMeasurement fields: acc, avel
            m.linear_acceleration = Vector3(
                x=float(e.payload.acc[0]),
                y=float(e.payload.acc[1]),
                z=float(e.payload.acc[2]),
            )
            m.angular_velocity = Vector3(
                x=float(e.payload.avel[0]),
                y=float(e.payload.avel[1]),
                z=float(e.payload.avel[2]),
            )
            writer.write(args.topic_imu, serialize_message(m), stamp_ns)

        elif e.kind == "gnss":
            # z is NED position from generator -> convert to lat/lon/h
            n = float(e.payload.pos[0]) if hasattr(e.payload, "pos") else float(e.payload.to_array()[0])
            ee = float(e.payload.pos[1]) if hasattr(e.payload, "pos") else float(e.payload.to_array()[1])
            d = float(e.payload.pos[2]) if hasattr(e.payload, "pos") else float(e.payload.to_array()[2])

            lat, lon, h = ned_to_lla(n, ee, d, args.datum_lat, args.datum_lon, args.datum_h)

            m = GNSSMsg()
            m.header = Header()
            m.header.stamp = stamp
            m.header.frame_id = "gnss_link"

            m.lat = float(lat)
            m.lon = float(lon)
            m.height = float(h)

            # Required by your callback gate
            m.fix_type = 3
            m.gnss_fix_ok = True

            # TODO: make sure fgo and message definition agree on accuracy units (mm vs m)
            # # FGO expects mm then divides by 1000.0
            # m.h_acc = int(args.h_acc_mm)
            # if hasattr(m, "v_acc"):
            #     m.v_acc = int(args.v_acc_mm)

            # keep units in meters to match message definition in your installed interface
            m.h_acc = float(args.h_acc_mm) / 1000.0
            if hasattr(m, "v_acc"):
                m.v_acc = float(args.v_acc_mm) / 1000.0

            writer.write(args.topic_gnss, serialize_message(m), stamp_ns)

        elif e.kind == "usbl":
            # z: [azimuth(rad), elevation(rad)]
            # az_rad = float(e.payload.z[0]) if hasattr(e.payload, "z") else float(e.payload.to_array()[0])
            # el_rad = float(e.payload.z[1]) if hasattr(e.payload, "z") else float(e.payload.to_array()[1])
            usbl_arr = meas_as_array(e.payload)
            az_rad = float(usbl_arr[0])
            el_rad = float(usbl_arr[1])

            rng = nearest(range_times, range_vals, e.t)
            dep = nearest(depth_times, depth_vals, e.t)

            # rng_m = float(rng.z[0]) if hasattr(rng, "z") else float(rng.to_array()[0])
            # dep_m = float(dep.z[0]) if hasattr(dep, "z") else float(dep.to_array()[0])
            rng_arr = meas_as_array(rng)
            dep_arr = meas_as_array(dep)
            rng_m = float(rng_arr[0])
            dep_m = float(dep_arr[0])

            t_sent_sec = float(e.t)
            tof_sec = max(rng_m / args.sound_speed, 0.0)
            t_recv_sec = t_sent_sec + tof_sec

            t_sent_us = int(round(t_sent_sec * 1e6))
            t_recv_us = int(round(t_recv_sec * 1e6))

            m = USBLMsg()
            m.header = Header()
            m.header.stamp = stamp
            m.header.frame_id = "usbl_link"

            m.rov_id = int(args.rov_id)
            m.azimuth = math.degrees(az_rad)     # FGO expects degrees
            m.elevation = math.degrees(el_rad)   # FGO expects degrees
            m.t_sent = t_sent_us
            m.t_received = t_recv_us

            # FGO uses msg->position.z for depth factor
            m.position = Vector3(x=0.0, y=0.0, z=dep_m)

            writer.write(args.topic_usbl, serialize_message(m), stamp_ns)

            if write_acoustic:
                a = AcousticMsg()
                a.header = Header()
                a.header.stamp = stamp
                a.header.frame_id = "acoustic_link"
                a.node_id = int(args.rov_id)
                a.t_sent = t_sent_us
                a.t_received = t_recv_us
                a.position = Point(x=0.0, y=0.0, z=dep_m)
                writer.write(args.topic_acoustic, serialize_message(a), stamp_ns)

    print(f"[OK] Wrote MCAP bag to: {args.out}")
    print("[OK] Topics:")
    print(f"  - {args.topic_imu}")
    print(f"  - {args.topic_gnss}")
    print(f"  - {args.topic_usbl}")
    if write_acoustic:
        print(f"  - {args.topic_acoustic}")

if __name__ == "__main__":
    rclpy.init()
    try:
        main()
    finally:
        rclpy.shutdown()