# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Export a recorded run to CSV, one row per state machine tick.

Each row holds the /fsm/state message and the latest detection, LiDAR and vehicle
values recorded before it, i.e. the same columns as state_log_v3.csv plus the vehicle
position and velocity. t_sim is the simulation time of the tick and t_rel the time since
the first non-zero raptor rig command (empty if the run has none).
Usage: ros2 run raptor_fsm export_run_csv <bag> <output.csv>
"""

import argparse
import csv

from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

FSM_TOPIC = '/fsm/state'
RIG_TOPIC = '/raptor_rig/cmd_vel'


def rounded(msg, getter, digits=3):
    return round(getter(msg), digits) if msg is not None else ''


def tick_row(t_sim, fsm, latest):
    det = latest.get('/perception/detection')
    lidar = latest.get('/lidar/hazard')
    mav = latest.get('/mavros/state')
    pose = latest.get('/mavros/local_position/pose')
    vel = latest.get('/mavros/local_position/velocity_local')
    rig = latest.get(RIG_TOPIC)
    return {
        't_sim': round(t_sim, 3),
        't_rel': '',  # filled in by export() once the rig start is known
        'state': fsm.state,
        'layer': fsm.layer_active,
        'fusion': round(fsm.fusion_score, 3),
        'proximity': round(fsm.proximity_score, 3),
        'yolo_conf': rounded(det, lambda m: m.max_conf),
        'yolo_hazard': rounded(det, lambda m: int(m.hazard), 0),
        'cx_norm': rounded(det, lambda m: m.cx_norm),
        'cy_norm': rounded(det, lambda m: m.cy_norm),
        'lidar_m': rounded(lidar, lambda m: m.min_range_m),
        'approach_mps': rounded(lidar, lambda m: m.approach_rate_mps),
        'sector': rounded(lidar, lambda m: m.sector_blocked, 0),
        'mode': mav.mode if mav is not None else '',
        'armed': rounded(mav, lambda m: int(m.armed), 0),
        'x_east': rounded(pose, lambda m: m.pose.position.x, 2),
        'y_north': rounded(pose, lambda m: m.pose.position.y, 2),
        'z_up': rounded(pose, lambda m: m.pose.position.z, 2),
        'v_east': rounded(vel, lambda m: m.twist.linear.x, 2),
        'v_north': rounded(vel, lambda m: m.twist.linear.y, 2),
        'v_up': rounded(vel, lambda m: m.twist.linear.z, 2),
        'rig_cmd_vx': rounded(rig, lambda m: m.linear.x, 2),
        'command': fsm.command,
        'note': fsm.note,
    }


def export(bag_path: str, output: str) -> int:
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=''),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    latest = {}
    rows = []
    t_sim = 0.0
    t_start = None  # simulation time of the first non-zero rig command
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, types[topic])
        if topic == '/clock':
            t_sim = msg.clock.sec + msg.clock.nanosec * 1e-9
        elif topic == FSM_TOPIC:
            rows.append(tick_row(t_sim, msg, latest))
        else:
            latest[topic] = msg
            # The bag stamps are wall-clock time; the latest /clock message is the
            # simulation time at which the command was recorded.
            if topic == RIG_TOPIC and t_start is None and msg.linear.x != 0.0:
                t_start = t_sim
    if not rows:
        raise SystemExit(f'no {FSM_TOPIC} messages in {bag_path}')
    if t_start is not None:
        for row in rows:
            row['t_rel'] = round(row['t_sim'] - t_start, 3)
    with open(output, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('bag', help='rosbag2 folder recorded by raptor.launch.py')
    parser.add_argument('output', help='CSV file to write')
    options = parser.parse_args(args)
    count = export(options.bag, options.output)
    print(f'{count} ticks -> {options.output}')


if __name__ == '__main__':
    main()
