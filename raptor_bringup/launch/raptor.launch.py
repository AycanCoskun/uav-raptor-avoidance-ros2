# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""Full raptor avoidance system in simulation.

Starts Gazebo with the sensor bridge (sim.launch.py), the LiDAR hazard node, the bird
detector, the state machine and MAVROS; optionally a test scenario and a rosbag recording.
ArduPilot SITL is started separately (see README):

  sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --out udp:127.0.0.1:14551
  ros2 launch raptor_bringup raptor.launch.py scenario:=raptor record:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, LaunchConfigurationNotEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

RECORDED_TOPICS = [
    '/fsm/state', '/perception/detection', '/lidar/hazard',
    '/mavros/state', '/mavros/local_position/pose', '/mavros/local_position/velocity_local',
    '/mavros/setpoint_raw/local', '/raptor_rig/cmd_vel', '/clock',
]


def generate_launch_description():
    pkg_share = get_package_share_directory('raptor_bringup')
    params = os.path.join(pkg_share, 'config', 'raptor_params.yaml')
    mavros_launch = os.path.join(get_package_share_directory('mavros'), 'launch', 'apm.launch')

    rviz = LaunchConfiguration('rviz')
    mavros = LaunchConfiguration('mavros')
    fcu_url = LaunchConfiguration('fcu_url')
    scenario = LaunchConfiguration('scenario')
    record = LaunchConfiguration('record')
    demo_camera = LaunchConfiguration('demo_camera')
    bag_path = LaunchConfiguration('bag_path')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false', description='Start RViz2.'),
        DeclareLaunchArgument('mavros', default_value='true', description='Start MAVROS.'),
        DeclareLaunchArgument(
            'fcu_url', default_value='udp://:14551@',
            description='MAVROS connection to ArduPilot SITL (MAVProxy --out port).'),
        DeclareLaunchArgument(
            'scenario', default_value='none',
            description='Test scenario: none, raptor or obstacle.'),
        DeclareLaunchArgument(
            'record', default_value='false', description='Record a rosbag of the run.'),
        DeclareLaunchArgument(
            'bag_path', default_value='raptor_run',
            description='Output folder of the rosbag (must not exist yet).'),
        DeclareLaunchArgument(
            'demo_camera', default_value='false',
            description='Spawn the third-person camera used for the demo recording.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'sim.launch.py')),
            launch_arguments={'rviz': rviz}.items()),

        Node(package='raptor_lidar', executable='lidar_hazard_node', name='lidar_hazard_node',
             output='screen', parameters=[params]),
        Node(package='raptor_perception', executable='bird_detector_node',
             name='bird_detector_node', output='screen', parameters=[params]),
        Node(package='raptor_fsm', executable='fsm_node', name='fsm_node',
             output='screen', parameters=[params]),

        IncludeLaunchDescription(
            XMLLaunchDescriptionSource(mavros_launch),
            launch_arguments={'fcu_url': fcu_url}.items(),
            condition=IfCondition(mavros)),

        Node(package='raptor_fsm', executable='scenario_node', name='scenario_node',
             output='screen', parameters=[params, {'scenario': scenario}],
             condition=LaunchConfigurationNotEquals('scenario', 'none')),

        # South-west of the scene looking north-east: the rig enters from the right and
        # the escape (north-west) goes to the left of the image.
        Node(package='ros_gz_sim', executable='create', name='spawn_demo_camera',
             output='screen',
             arguments=['-world', 'raptor_runway', '-name', 'demo_camera',
                        '-file', os.path.join(pkg_share, 'models', 'demo_camera', 'model.sdf'),
                        '-x', '-20.0', '-y', '-10.0', '-z', '13.0', '-P', '0.15', '-Y', '0.84'],
             condition=IfCondition(demo_camera)),

        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', bag_path] + RECORDED_TOPICS,
            output='screen', condition=IfCondition(record)),
    ])
