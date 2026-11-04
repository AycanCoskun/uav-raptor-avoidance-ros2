# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Scenario runner for the full-loop simulation tests.

Simulation counterpart of the staged-target tests in the paper.

scenario:=raptor    take off, place the raptor rig next to the drone and drive it towards
                    the drone. Expected: IDLE -> EARLY_WARNING -> THREAT_CONFIRMED ->
                    ESCAPE -> RTL.
scenario:=obstacle  take off and fly slowly towards the test pillar; stop sending
                    commands 1.6 m before it (ArduPilot keeps the last velocity), so
                    the state machine alone takes over at 1.2 m. Expected: IDLE ->
                    OBSTACLE_AVOID -> IDLE.

Every state machine transition is logged with its simulation time.
"""

import subprocess
import threading
import time

from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from raptor_interfaces.msg import FsmState, LidarHazard
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

VELOCITY_MASK = (
    PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ
    | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ
    | PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE)
EXPECTED = {
    'raptor': ['EARLY_WARNING', 'THREAT_CONFIRMED', 'ESCAPE', 'RTL'],
    'obstacle': ['OBSTACLE_AVOID', 'IDLE'],
}


class ScenarioError(RuntimeError):
    pass


class ScenarioNode(Node):

    def __init__(self):
        super().__init__('scenario_node')
        self.scenario = self.declare_parameter('scenario', 'raptor').value
        self.takeoff_enabled = self.declare_parameter('takeoff', True).value
        self.takeoff_alt_m = self.declare_parameter('takeoff_alt_m', 10.0).value
        self.settle_s = self.declare_parameter('settle_s', 3.0).value
        self.world = self.declare_parameter('world', 'raptor_runway').value
        self.rig_model = self.declare_parameter('rig_model', 'raptor_rig').value
        self.start_distance_m = self.declare_parameter('start_distance_m', 4.0).value
        self.stop_distance_m = self.declare_parameter('stop_distance_m', 0.8).value
        self.approach_speed_mps = self.declare_parameter('approach_speed_mps', 0.5).value
        self.bird_height_m = self.declare_parameter('bird_height_m', 2.0).value
        self.bird_offset_m = self.declare_parameter('bird_offset_m', 0.3).value
        self.camera_z_offset_m = self.declare_parameter('camera_z_offset_m', 0.08).value
        self.camera_y_offset_m = self.declare_parameter('camera_y_offset_m', 0.06).value
        self.forward_speed_mps = self.declare_parameter('forward_speed_mps', 0.2).value
        self.release_distance_m = self.declare_parameter('release_distance_m', 1.6).value
        self.monitor_s = self.declare_parameter('monitor_s', 30.0).value

        self._lock = threading.Lock()
        self._mav_state = State()
        self._pose = None
        self._fsm_state = None
        self._lidar_range = float('inf')
        self._transitions = []
        self._t0 = None

        self.create_subscription(State, 'mavros/state', self._on_state, 10)
        self.create_subscription(
            PoseStamped, 'mavros/local_position/pose', self._on_pose, qos_profile_sensor_data)
        self.create_subscription(FsmState, 'fsm/state', self._on_fsm, 10)
        self.create_subscription(LidarHazard, 'lidar/hazard', self._on_lidar, 10)
        self._rig_pub = self.create_publisher(Twist, 'raptor_rig/cmd_vel', 10)
        self._setpoint_pub = self.create_publisher(
            PositionTarget, 'mavros/setpoint_raw/local', 10)
        self._set_mode = self.create_client(SetMode, 'mavros/set_mode')
        self._arming = self.create_client(CommandBool, 'mavros/cmd/arming')
        self._takeoff = self.create_client(CommandTOL, 'mavros/cmd/takeoff')

    # Callbacks

    def _on_state(self, msg: State) -> None:
        with self._lock:
            self._mav_state = msg

    def _on_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            self._pose = msg.pose.position

    def _on_lidar(self, msg: LidarHazard) -> None:
        with self._lock:
            self._lidar_range = msg.min_range_m

    def _on_fsm(self, msg: FsmState) -> None:
        with self._lock:
            if self._fsm_state is not None and msg.state != self._fsm_state:
                t = self.now_s() - self._t0 if self._t0 is not None else 0.0
                self._transitions.append(
                    (t, self._fsm_state, msg.state, msg.fusion_score, msg.command, msg.note))
                self.get_logger().info(
                    f't={t:6.2f} s  {self._fsm_state} -> {msg.state}  '
                    f'fusion={msg.fusion_score:.3f}  cmd={msg.command}  {msg.note}')
            self._fsm_state = msg.state

    # Helpers

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def mav_state(self) -> State:
        with self._lock:
            return self._mav_state

    def pose(self):
        with self._lock:
            return self._pose

    def fsm_state(self):
        with self._lock:
            return self._fsm_state

    def lidar_range(self) -> float:
        with self._lock:
            return self._lidar_range

    def wait_for(self, predicate, timeout_s: float, what: str) -> None:
        end = time.monotonic() + timeout_s
        while not predicate():
            if time.monotonic() > end:
                raise ScenarioError(f'timed out waiting for {what}')
            time.sleep(0.1)

    def sim_sleep(self, seconds: float) -> None:
        end = self.now_s() + seconds
        while self.now_s() < end:
            time.sleep(0.02)

    def call(self, client, request, timeout_s: float = 10.0):
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise ScenarioError(f'service {client.srv_name} not available')
        future = client.call_async(request)
        self.wait_for(future.done, timeout_s, client.srv_name)
        return future.result()

    def set_mode(self, mode: str) -> None:
        for _ in range(10):
            if self.mav_state().mode == mode:
                return
            self.call(self._set_mode, SetMode.Request(custom_mode=mode))
            time.sleep(1.0)
        raise ScenarioError(f'could not set mode {mode}')

    # Steps

    def take_off(self) -> None:
        self.get_logger().info('waiting for MAVROS connection and position')
        self.wait_for(lambda: self.mav_state().connected, 120.0, 'FCU connection')
        self.wait_for(lambda: self.pose() is not None, 60.0, 'local position')
        if self.mav_state().armed and self.pose().z > self.takeoff_alt_m - 1.0:
            self.get_logger().info('already airborne, skipping take-off')
            return
        self.set_mode('GUIDED')
        # Pre-arm checks (EKF, GPS) can take a while after SITL starts: retry arming.
        end = time.monotonic() + 180.0
        while not self.mav_state().armed:
            if time.monotonic() > end:
                raise ScenarioError('could not arm')
            self.call(self._arming, CommandBool.Request(value=True))
            time.sleep(2.0)
        # Take off right after arming, otherwise ArduPilot disarms again.
        result = self.call(self._takeoff, CommandTOL.Request(altitude=float(self.takeoff_alt_m)))
        if not result.success:
            raise ScenarioError('take-off rejected')
        self.get_logger().info(f'taking off to {self.takeoff_alt_m:.1f} m')
        self.wait_for(lambda: self.pose().z > self.takeoff_alt_m - 0.5, 90.0, 'take-off altitude')
        self.sim_sleep(self.settle_s)

    def set_rig_pose(self, x: float, y: float, z: float) -> None:
        request = (f'name: "{self.rig_model}", position: {{x: {x}, y: {y}, z: {z}}}, '
                   f'orientation: {{w: 1}}')
        result = subprocess.run(
            ['gz', 'service', '-s', f'/world/{self.world}/set_pose',
             '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '3000', '--req', request],
            capture_output=True, text=True, check=False)
        if 'true' not in result.stdout:
            raise ScenarioError(f'set_pose failed for {self.rig_model}: {result.stderr}')

    def drive_rig(self, speed_x: float, duration_s: float) -> None:
        twist = Twist()
        twist.linear.x = float(speed_x)
        end = self.now_s() + duration_s
        while self.now_s() < end:
            self._rig_pub.publish(twist)
            self.sim_sleep(0.1)
        for _ in range(5):
            self._rig_pub.publish(Twist())
            self.sim_sleep(0.1)

    def run_raptor(self) -> None:
        # Gazebo world = MAVROS local ENU here (the drone spawns at the origin; local z is
        # base_link height, 0.19 m on the ground); the camera is 0.08 m above base_link.
        p = self.pose()
        x = p.x + self.start_distance_m - self.bird_offset_m
        y = p.y + self.camera_y_offset_m
        z = p.z + self.camera_z_offset_m + self.bird_height_m
        self.get_logger().info(
            f'raptor rig: pole {self.start_distance_m:.1f} m east of the drone, bird '
            f'{self.bird_height_m:.1f} m above the camera, approach '
            f'{self.approach_speed_mps:.2f} m/s to {self.stop_distance_m:.1f} m')
        self.set_rig_pose(x, y, z)
        self.sim_sleep(1.0)
        self._t0 = self.now_s()
        travel = self.start_distance_m - self.stop_distance_m
        self.drive_rig(-self.approach_speed_mps, travel / self.approach_speed_mps)
        end = self.now_s() + self.monitor_s
        while self.fsm_state() != 'RTL' and self.now_s() < end:
            time.sleep(0.1)

    def run_obstacle(self) -> None:
        self.get_logger().info(
            f'flying north at {self.forward_speed_mps:.2f} m/s, releasing control '
            f'{self.release_distance_m:.1f} m before the pillar')
        self.set_mode('GUIDED')
        self._t0 = self.now_s()
        msg = PositionTarget()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = VELOCITY_MASK
        msg.velocity.y = float(self.forward_speed_mps)   # ENU: +y = north
        end = self._t0 + self.monitor_s
        # Only one node commands the vehicle at a time: stop before the FSM can react.
        while (self.fsm_state() == 'IDLE' and self.lidar_range() > self.release_distance_m
               and self.now_s() < end):
            msg.header.stamp = self.get_clock().now().to_msg()
            self._setpoint_pub.publish(msg)
            self.sim_sleep(0.1)
        # The state machine is in control now; wait until it is back in IDLE for 2 s.
        self.wait_for(lambda: self.fsm_state() != 'IDLE' or self.now_s() > end, 60.0,
                      'the state machine to take over')
        while self.now_s() < end:
            if self.fsm_state() == 'IDLE':
                self.sim_sleep(2.0)
                if self.fsm_state() == 'IDLE':
                    break
            time.sleep(0.1)

    def run(self) -> bool:
        if self.scenario not in ('raptor', 'obstacle'):
            raise ScenarioError(f'unknown scenario {self.scenario}')
        if self.takeoff_enabled:
            self.take_off()
        self.wait_for(lambda: self.fsm_state() is not None, 30.0, '/fsm/state')
        if self.scenario == 'raptor':
            self.run_raptor()
        else:
            self.run_obstacle()
        self.get_logger().info('transitions (t from scenario start):')
        for t, old, new, fusion, command, note in self._transitions:
            self.get_logger().info(
                f'  {t:6.2f} s  {old} -> {new}  fusion={fusion:.3f}  {command}  {note}')
        # PASS when the expected states were entered in this order.
        visited = iter(new for _, _, new, _, _, _ in self._transitions)
        passed = all(state in visited for state in EXPECTED[self.scenario])
        self.get_logger().info(f'scenario {self.scenario}: {"PASS" if passed else "FAIL"}')
        return passed


def main(args=None):
    rclpy.init(args=args)
    node = ScenarioNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    code = 1
    try:
        code = 0 if node.run() else 1
    except ScenarioError as error:
        node.get_logger().error(str(error))
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    main()
