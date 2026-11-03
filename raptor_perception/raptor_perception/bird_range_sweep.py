# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Detection range sweep in Gazebo (simulation test tool).

Moves the bird_target model to a list of heights above the upward-facing camera
with the /world/<world>/set_pose service and records what bird_detector_node
publishes on /perception/detection at each distance. Writes one CSV row per
distance. Needs the simulation (sim.launch.py) and bird_detector_node running.
"""

import csv
import math
import statistics
import subprocess
import time

from raptor_interfaces.msg import VisionDetection
import rclpy
from rclpy.node import Node

DEFAULT_DISTANCES = [
    1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]


class BirdRangeSweep(Node):

    def __init__(self):
        super().__init__('bird_range_sweep')
        self.world = self.declare_parameter('world', 'raptor_runway').value
        self.model = self.declare_parameter('model', 'bird_target').value
        self.x = self.declare_parameter('x', 0.6).value
        self.y = self.declare_parameter('y', 0.06).value
        self.camera_z = self.declare_parameter('camera_z', 0.28).value
        self.yaw_deg = self.declare_parameter('yaw_deg', 0.0).value
        self.distances = self.declare_parameter('distances', DEFAULT_DISTANCES).value
        self.settle_s = self.declare_parameter('settle_s', 2.0).value
        self.samples = self.declare_parameter('samples', 8).value
        self.conf_threshold = self.declare_parameter('conf_threshold', 0.35).value
        self.output = self.declare_parameter('output', 'bird_range_sweep.csv').value
        self._collect = False
        self._buffer = []
        self.create_subscription(
            VisionDetection, 'perception/detection', self._on_detection, 10)

    def _on_detection(self, msg: VisionDetection) -> None:
        if self._collect:
            self._buffer.append(msg)

    def move_bird(self, z: float) -> bool:
        half_yaw = math.radians(self.yaw_deg) / 2.0
        request = (f'name: "{self.model}", '
                   f'position: {{x: {self.x}, y: {self.y}, z: {z}}}, '
                   f'orientation: {{z: {math.sin(half_yaw)}, w: {math.cos(half_yaw)}}}')
        result = subprocess.run(
            ['gz', 'service', '-s', f'/world/{self.world}/set_pose',
             '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '3000', '--req', request],
            capture_output=True, text=True, check=False)
        return 'true' in result.stdout

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def measure(self, distance: float) -> dict:
        if not self.move_bird(self.camera_z + distance):
            raise RuntimeError(f'set_pose failed for {self.model} in world {self.world}')
        self.spin_for(self.settle_s)
        self._buffer = []
        self._collect = True
        deadline = time.monotonic() + 30.0
        while len(self._buffer) < self.samples and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        self._collect = False
        msgs = self._buffer[:self.samples]
        if not msgs:
            raise RuntimeError('no messages on /perception/detection')
        confs = [m.max_conf for m in msgs]
        hits = [m for m in msgs if m.hazard]
        return {
            'distance_m': distance,
            'yaw_deg': self.yaw_deg,
            'frames': len(msgs),
            'conf_mean': round(statistics.mean(confs), 3),
            'conf_min': round(min(confs), 3),
            'conf_max': round(max(confs), 3),
            'detect_rate': round(len(hits) / len(msgs), 2),
            'confirmed_rate': round(sum(m.confirmed for m in msgs) / len(msgs), 2),
            'cx_norm': round(statistics.mean(m.cx_norm for m in hits), 3) if hits else '',
            'cy_norm': round(statistics.mean(m.cy_norm for m in hits), 3) if hits else '',
        }

    def run(self) -> None:
        rows = []
        for distance in self.distances:
            row = self.measure(float(distance))
            rows.append(row)
            self.get_logger().info(
                f"d={row['distance_m']:.1f} m  conf mean={row['conf_mean']:.2f} "
                f"max={row['conf_max']:.2f}  detected={row['detect_rate']:.0%}  "
                f"confirmed={row['confirmed_rate']:.0%}")
        with open(self.output, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.get_logger().info(f'wrote {self.output}')


def main(args=None):
    rclpy.init(args=args)
    node = BirdRangeSweep()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
