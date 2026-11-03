# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
State machine node that commands ArduPilot through MAVROS.

Fuses /perception/detection and /lidar/hazard at 4 Hz. Replaces the CSV tail threads,
the main loop and the pymavlink calls of state_machine_v3.py. Publishes one
raptor_interfaces/FsmState per tick (replaces state_log_v3.csv).
"""

from collections import deque
import dataclasses
import math

from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import SetMode
from raptor_fsm.fsm_core import FsmInputs, FsmParams, RaptorFsm
from raptor_interfaces.msg import FsmState, LidarHazard, VisionDetection
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

# Velocity only: ignore position, acceleration, yaw and yaw rate (0b0000111111000111 in v3).
VELOCITY_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ
    | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ
    | PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE)
MODE_RETRIES = 20


class FsmNode(Node):

    def __init__(self):
        super().__init__('fsm_node')
        defaults = FsmParams()
        params = FsmParams(**{
            f.name: float(self.declare_parameter(f.name, getattr(defaults, f.name)).value)
            for f in dataclasses.fields(FsmParams)})
        loop_hz = self.declare_parameter('loop_hz', 4.0).value
        self._input_timeout_s = self.declare_parameter('input_timeout_s', 1.0).value
        vision_window = self.declare_parameter('vision_window', 3).value
        self._dry_run = self.declare_parameter('dry_run', False).value

        self._fsm = RaptorFsm(params)
        # Like v3, look at the last few detections so a brief one is not missed between ticks.
        self._vision = deque(maxlen=vision_window)
        self._lidar = None
        self._lidar_time = None
        self._mav_state = State()
        self._pending_mode = None
        self._mode_attempts = 0
        self._mode_future = None
        self._last_command = ''

        self.create_subscription(VisionDetection, 'perception/detection', self._on_vision, 10)
        self.create_subscription(LidarHazard, 'lidar/hazard', self._on_lidar, 10)
        self.create_subscription(State, 'mavros/state', self._on_mav_state, 10)
        self._state_pub = self.create_publisher(FsmState, 'fsm/state', 10)
        self._setpoint_pub = self.create_publisher(
            PositionTarget, 'mavros/setpoint_raw/local', 10)
        self._set_mode = self.create_client(SetMode, 'mavros/set_mode')
        self.create_timer(1.0 / loop_hz, self._tick)

        self.get_logger().info(
            f'loop_hz={loop_hz} fusion_warn={params.fusion_warn} '
            f'fusion_confirm={params.fusion_confirm} d_warn={params.d_warn_m} m '
            f'd_hazard={params.d_hazard_m} m d_confirm={params.d_confirm_m} m '
            f'escape={params.escape_duration_s} s dry_run={self._dry_run}')

    # Inputs

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_vision(self, msg: VisionDetection) -> None:
        self._vision.append((self._now_s(), msg))

    def _on_lidar(self, msg: LidarHazard) -> None:
        self._lidar = msg
        self._lidar_time = self._now_s()

    def _on_mav_state(self, msg: State) -> None:
        self._mav_state = msg

    def _snapshot(self, now_s: float) -> FsmInputs:
        """Latest sensor data; anything older than input_timeout_s counts as no detection."""
        inputs = FsmInputs()
        fresh = [m for t, m in self._vision if now_s - t <= self._input_timeout_s]
        if fresh:
            best = max(fresh, key=lambda m: m.max_conf)
            inputs.yolo_hazard = any(m.hazard for m in fresh)
            inputs.yolo_confirmed = any(m.confirmed for m in fresh)
            inputs.yolo_conf = float(best.max_conf)
            inputs.cx_norm = float(best.cx_norm)
            inputs.cy_norm = float(best.cy_norm)
        if self._lidar is not None and now_s - self._lidar_time <= self._input_timeout_s:
            inputs.lidar_hazard = bool(self._lidar.hazard)
            inputs.lidar_confirmed = bool(self._lidar.confirmed)
            inputs.distance_m = float(self._lidar.min_range_m)
            inputs.approach_rate_mps = float(self._lidar.approach_rate_mps)
            inputs.sector = int(self._lidar.sector_blocked)
        return inputs

    # Tick

    def _tick(self) -> None:
        now_s = self._now_s()
        inputs = self._snapshot(now_s)
        result = self._fsm.step(inputs, now_s)

        command = result.command
        if command.mode is not None:
            self._request_mode(command.mode)
        if command.velocity_ned is not None:
            self._send_velocity(command.velocity_ned)
        if command.label:
            self._last_command = command.label
            if command.velocity_ned is not None:
                vx, vy, vz = command.velocity_ned
                self._last_command += f' vel_ned=({vx:.2f},{vy:.2f},{vz:.2f})'
        self._service_mode_request()

        msg = FsmState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.state = result.state
        msg.layer_active = result.layer_active
        msg.fusion_score = float(result.fusion_score)
        msg.proximity_score = float(result.proximity_score)
        msg.command = self._last_command
        msg.note = result.note
        self._state_pub.publish(msg)

        if result.changed:
            distance = inputs.distance_m if math.isfinite(inputs.distance_m) else -1.0
            self.get_logger().info(
                f'{result.previous_state} -> {result.state} | layer={result.layer_active} '
                f'fusion={result.fusion_score:.3f} yolo={inputs.yolo_conf:.2f} '
                f'd={distance:.2f} m rate={inputs.approach_rate_mps:.2f} m/s | '
                f'cmd={command.label or "-"} | {result.note}')

    # MAVROS

    def _send_velocity(self, velocity_ned) -> None:
        if self._dry_run:
            return
        vx, vy, vz = velocity_ned
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = VELOCITY_MASK
        # MAVROS takes ENU and converts to NED itself: east = vy, north = vx, up = -vz.
        msg.velocity.x = float(vy)
        msg.velocity.y = float(vx)
        msg.velocity.z = float(-vz)
        self._setpoint_pub.publish(msg)

    def _request_mode(self, mode: str) -> None:
        if self._dry_run or self._mav_state.mode == mode:
            return
        self._pending_mode = mode
        self._mode_attempts = 0

    def _service_mode_request(self) -> None:
        """Send the pending mode change, retrying each tick until /mavros/state shows it."""
        if self._pending_mode is None:
            return
        if self._mav_state.mode == self._pending_mode:
            self._pending_mode = None
            return
        if self._mode_future is not None and not self._mode_future.done():
            return
        if self._mode_attempts >= MODE_RETRIES:
            self.get_logger().error(f'Could not set mode {self._pending_mode}')
            self._pending_mode = None
            return
        if not self._set_mode.service_is_ready():
            self.get_logger().warning(
                'mavros/set_mode not available', throttle_duration_sec=5.0)
            self._mode_attempts += 1
            return
        self._mode_attempts += 1
        self._mode_future = self._set_mode.call_async(
            SetMode.Request(custom_mode=self._pending_mode))


def main(args=None):
    rclpy.init(args=args)
    node = FsmNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
