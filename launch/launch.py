#!/usr/bin/env python3
import os
import launch
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from ament_index_python import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():

    # Declare a launch argument to specify the path to the xacro file
    xacro_file_arg = DeclareLaunchArgument(
        'xacro_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('car'),
            'urdf',
            'car.xacro'
        ]),
        description='Path to the xacro file for the robot'
    )

    # Get the xacro file path from the launch argument
    xacro_file = LaunchConfiguration('xacro_file')

    # Use Command substitution to run xacro and get the robot description
    robot_description = Command(xacro_file)

    return LaunchDescription([
        xacro_file_arg,  # Declare the xacro file argument

        # Launch the robot_state_publisher node
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),

        # Launch the joint_state_publisher node
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher'
        ),

        # Launch RViz2 with the specified configuration file
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(
                get_package_share_directory('car'),
                'rviz', 
                'robot.rviz'
            )]
        )
    ])
