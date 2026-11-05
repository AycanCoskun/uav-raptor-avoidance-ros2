# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""Start Gazebo Harmonic with the raptor test world and bridge sensor topics to ROS 2.

ArduPilot SITL is started separately (see README) and connects to the
ArduPilotPlugin inside the iris_raptor model.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('raptor_bringup')
    gz_sim_launch = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')

    world = LaunchConfiguration('world')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(pkg_share, 'worlds', 'raptor_runway.sdf'),
            description='Full path to the Gazebo world file.'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Start RViz2 with the sensor view.'),

        # Let Gazebo find the iris_raptor model shipped with this package.
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH', os.path.join(pkg_share, 'models')),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_sim_launch),
            launch_arguments={'gz_args': ['-r -v 3 ', world]}.items()),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_bridge',
            output='screen',
            parameters=[{
                'config_file': os.path.join(pkg_share, 'config', 'gz_bridge.yaml'),
                'use_sim_time': True,
            }]),

        # Static sensor frames (poses match the joints in iris_raptor/model.sdf).
        # camera_link is pitched -90 deg so its +X axis (viewing direction) points up.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_lidar_tf',
            arguments=['--z', '0.10',
                       '--frame-id', 'base_link', '--child-frame-id', 'lidar_link']),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_tf',
            arguments=['--x', '0.06', '--z', '0.08', '--pitch', '-1.5708',
                       '--frame-id', 'base_link', '--child-frame-id', 'camera_link']),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(pkg_share, 'rviz', 'sensors.rviz')],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(rviz)),
    ])
