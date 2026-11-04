# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Record the demo GIF of a raptor scenario run.

Combines the third-person view (/demo_camera/image, launch with demo_camera:=true), the
detector debug image and the state machine output into one frame per demo camera image,
from shortly before the rig starts moving until a few seconds after RTL. The third-person
view is a 2.5x crop that follows the drone: its MAVROS position is projected into the demo
camera image (same pose as in raptor.launch.py). Frames are in simulation time, so the GIF
plays at simulation speed. Also saves a PNG still per state.
"""

from collections import deque
import math
import os

import cv2
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from PIL import Image as PilImage
from raptor_interfaces.msg import FsmState, LidarHazard, VisionDetection
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

STATE_COLOURS = {  # RGB
    'IDLE': (110, 110, 110),
    'OBSTACLE_AVOID': (40, 120, 220),
    'EARLY_WARNING': (230, 170, 0),
    'THREAT_CONFIRMED': (235, 100, 20),
    'ESCAPE': (210, 30, 30),
    'RTL': (30, 140, 70),
}


def to_rgb(msg: Image):
    """Return an RGB uint8 array for rgb8 / bgr8 images, None otherwise."""
    if msg.encoding not in ('rgb8', 'bgr8'):
        return None
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return np.ascontiguousarray(image[:, :, ::-1] if msg.encoding == 'bgr8' else image)


class DemoRecorder(Node):

    def __init__(self):
        super().__init__('demo_recorder')
        self.output_dir = self.declare_parameter('output_dir', 'demo').value
        self.height = self.declare_parameter('panel_height', 315).value
        self.pre_roll_s = self.declare_parameter('pre_roll_s', 1.0).value
        self.post_roll_s = self.declare_parameter('post_roll_s', 2.5).value
        self.colours = self.declare_parameter('gif_colours', 64).value
        # Demo camera pose and field of view (must match raptor.launch.py / model.sdf).
        self.cam_pos = self.declare_parameter('camera_xyz', [-20.0, -10.0, 13.0]).value
        self.cam_pitch = self.declare_parameter('camera_pitch', 0.15).value
        self.cam_yaw = self.declare_parameter('camera_yaw', 0.84).value
        self.cam_hfov = self.declare_parameter('camera_hfov', 1.05).value
        self.zoom = self.declare_parameter('zoom', 2.5).value
        self.max_duration_s = self.declare_parameter('max_duration_s', 40.0).value
        self.gif_step = self.declare_parameter('gif_frame_step', 1).value

        self._debug = None
        self._fsm = None
        self._detection = None
        self._lidar = None
        self._drone = None
        self._crop_centre = None
        self._pre_roll = deque()
        self._frames = []        # (t_sim, PIL image)
        self._stills = {}        # state -> PIL image (first frame after entering it)
        self._start = None
        self._end = None
        self.done = False

        self.create_subscription(
            Image, 'demo_camera/image', self._on_demo, qos_profile_sensor_data)
        self.create_subscription(Image, 'perception/debug_image', self._on_debug, 10)
        self.create_subscription(FsmState, 'fsm/state', self._on_fsm, 10)
        self.create_subscription(VisionDetection, 'perception/detection', self._on_det, 10)
        self.create_subscription(LidarHazard, 'lidar/hazard', self._on_lidar, 10)
        self.create_subscription(Twist, 'raptor_rig/cmd_vel', self._on_rig, 10)
        self.create_subscription(
            PoseStamped, 'mavros/local_position/pose', self._on_pose, qos_profile_sensor_data)
        self.get_logger().info('waiting for the raptor rig to move')

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_debug(self, msg):
        self._debug = to_rgb(msg)

    def _on_fsm(self, msg):
        self._fsm = msg
        if msg.state == 'RTL' and self._start is not None and self._end is None:
            self._end = self._now_s() + self.post_roll_s

    def _on_det(self, msg):
        self._detection = msg

    def _on_lidar(self, msg):
        self._lidar = msg

    def _on_pose(self, msg):
        # MAVROS local ENU equals the Gazebo world frame here (spawn at the origin).
        p = msg.pose.position
        self._drone = (p.x, p.y, p.z)

    def _on_rig(self, msg):
        if self._start is None and msg.linear.x != 0.0:
            self._start = self._now_s()
            self._frames = list(self._pre_roll)
            self.get_logger().info('rig moving: recording')

    def _on_demo(self, msg):
        if self.done:
            return
        view = to_rgb(msg)
        if view is None:
            return
        now = self._now_s()
        frame = self._compose(view, now)
        if self._start is None:
            self._pre_roll.append((now, frame))
            while self._pre_roll and now - self._pre_roll[0][0] > self.pre_roll_s:
                self._pre_roll.popleft()
            return
        self._frames.append((now, frame))
        state = self._fsm.state if self._fsm is not None else ''
        if state and state not in self._stills:
            self._stills[state] = frame
        if (self._end is not None and now >= self._end) or now - self._start > self.max_duration_s:
            self.done = True

    def _project(self, point, width, height):
        """Pixel of a world point in the demo camera (x forward, y left, z up)."""
        dx, dy, dz = (point[i] - self.cam_pos[i] for i in range(3))
        cy, sy = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
        cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        x1, y1 = cy * dx + sy * dy, -sy * dx + cy * dy
        forward, left, up = cp * x1 - sp * dz, y1, sp * x1 + cp * dz
        if forward <= 0.1:
            return None
        focal = width / 2.0 / math.tan(self.cam_hfov / 2.0)
        return (width / 2.0 - focal * left / forward, height / 2.0 - focal * up / forward)

    def _track(self, view):
        """Crop around the drone, smoothed so the virtual camera pans steadily."""
        height, width = view.shape[:2]
        crop_w, crop_h = int(width / self.zoom), int(height / self.zoom)
        target = self._project(self._drone, width, height) if self._drone else None
        if target is None:
            target = (width / 2.0, height / 2.0)
        if self._crop_centre is None:
            self._crop_centre = target
        self._crop_centre = tuple(0.7 * c + 0.3 * t for c, t in zip(self._crop_centre, target))
        x0 = int(min(max(self._crop_centre[0] - crop_w / 2, 0), width - crop_w))
        y0 = int(min(max(self._crop_centre[1] - crop_h / 2, 0), height - crop_h))
        return view[y0:y0 + crop_h, x0:x0 + crop_w]

    def _compose(self, view, now):
        h = self.height
        view = self._track(view)
        view = cv2.resize(view, (int(view.shape[1] * h / view.shape[0]), h),
                          interpolation=cv2.INTER_AREA)
        if self._debug is not None:
            debug_w = int(self._debug.shape[1] * h / self._debug.shape[0])
            debug = cv2.resize(self._debug, (debug_w, h), interpolation=cv2.INTER_AREA)
        else:
            debug = np.zeros((h, int(h * 4 / 3), 3), dtype=np.uint8)
        body = np.hstack([view, np.full((h, 4, 3), 255, np.uint8), debug])
        width = body.shape[1]

        state = self._fsm.state if self._fsm is not None else 'IDLE'
        banner = np.zeros((44, width, 3), np.uint8)
        banner[:] = STATE_COLOURS.get(state, (110, 110, 110))
        t = now - self._start if self._start is not None else 0.0
        fusion = self._fsm.fusion_score if self._fsm is not None else 0.0
        conf = self._detection.max_conf if self._detection is not None else 0.0
        dist = self._lidar.min_range_m if self._lidar is not None else float('inf')
        dist_text = f'{dist:.2f} m' if np.isfinite(dist) else '-'
        cv2.putText(banner, state, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        info = (f't={t:5.2f} s   YOLO conf={conf:.2f}   LiDAR d={dist_text}   '
                f'fusion={fusion:.2f}')
        cv2.putText(banner, info, (330, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        footer = np.full((24, width, 3), 30, np.uint8)
        cv2.putText(footer, 'Simulation (Gazebo Harmonic + ArduPilot SITL + ROS 2)  |  left: '
                    'third-person view  |  right: upward camera + YOLOv5n',
                    (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        return PilImage.fromarray(np.vstack([banner, body, footer]))

    def save(self) -> None:
        if not self._frames:
            self.get_logger().error('no frames recorded')
            return
        os.makedirs(self.output_dir, exist_ok=True)
        frames = [(t, f.quantize(colors=self.colours)) for t, f in self._frames[::self.gif_step]]
        durations = [max(20, int(round((b[0] - a[0]) * 1000)))
                     for a, b in zip(frames, frames[1:])] + [1500]
        path = os.path.join(self.output_dir, 'demo_raptor_scenario.gif')
        frames[0][1].save(path, save_all=True, append_images=[f for _, f in frames[1:]],
                          duration=durations, loop=0, optimize=True)
        for state, image in self._stills.items():  # stills keep full colour
            image.save(os.path.join(self.output_dir, f'still_{state.lower()}.png'))
        self.get_logger().info(
            f'{len(frames)} frames, {frames[-1][0] - frames[0][0]:.1f} s sim -> {path}')


def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorder()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.save()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.save()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
