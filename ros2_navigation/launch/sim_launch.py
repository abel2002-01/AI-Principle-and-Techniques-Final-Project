#!/usr/bin/env python3
"""
Launch Gazebo Sim + ros_gz_bridge + navigation node.

This is the recommended end-to-end demo for Question 5 on ROS 2 Jazzy.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("traveling_ethiopia_robot")
    world_file = os.path.join(pkg_share, "worlds", "ethiopia_cities.world")

    # Arguments
    initial_city_arg = DeclareLaunchArgument(
        "initial_city",
        default_value="Addis Ababa",
        description="Starting city for navigation",
    )
    goal_city_arg = DeclareLaunchArgument(
        "goal_city",
        default_value="Moyale",
        description="Destination city for navigation",
    )
    strategy_arg = DeclareLaunchArgument(
        "search_strategy",
        default_value="bfs",
        description="Search strategy (bfs or dfs)",
    )

    # Gazebo Sim
    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", world_file],
        output="screen",
    )

    # Bridge cmd_vel (ROS->GZ) and odometry (GZ->ROS)
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/model/three_wheel_robot/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/model/three_wheel_robot/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        ],
    )

    # Navigation node
    navigation_node = Node(
        package="traveling_ethiopia_robot",
        executable="navigation_node.py",
        name="navigation_node",
        output="screen",
        parameters=[
            {
                "initial_city": LaunchConfiguration("initial_city"),
                "goal_city": LaunchConfiguration("goal_city"),
                "search_strategy": LaunchConfiguration("search_strategy"),
                "cmd_vel_topic": "/model/three_wheel_robot/cmd_vel",
                "odom_topic": "/model/three_wheel_robot/odometry",
            }
        ],
    )

    return LaunchDescription(
        [
            initial_city_arg,
            goal_city_arg,
            strategy_arg,
            # Ensure Gazebo Sim can resolve assets relative to the world file
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH",
                value=os.path.join(pkg_share, "worlds"),
            ),
            gazebo,
            bridge,
            navigation_node,
        ]
    )
