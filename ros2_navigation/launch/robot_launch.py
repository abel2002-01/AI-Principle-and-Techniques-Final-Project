#!/usr/bin/env python3
"""
Launch file for the Three-Wheel Robot control in Gazebo Sim.

This launch file:
1. Starts ros_gz_bridge for cmd_vel and odometry
2. Starts the navigation node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    initial_city_arg = DeclareLaunchArgument(
        'initial_city',
        default_value='Addis Ababa',
        description='Starting city for navigation'
    )
    
    goal_city_arg = DeclareLaunchArgument(
        'goal_city',
        default_value='Moyale',
        description='Destination city for navigation'
    )
    
    strategy_arg = DeclareLaunchArgument(
        'search_strategy',
        default_value='bfs',
        description='Search strategy (bfs or dfs)'
    )
    
    # Bridge cmd_vel (ROS->GZ) and odometry (GZ->ROS)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/model/three_wheel_robot/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/model/three_wheel_robot/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        ],
    )
    
    # Navigation node
    navigation_node = Node(
        package='traveling_ethiopia_robot',
        executable='navigation_node.py',
        name='navigation_node',
        output='screen',
        parameters=[{
            'initial_city': LaunchConfiguration('initial_city'),
            'goal_city': LaunchConfiguration('goal_city'),
            'search_strategy': LaunchConfiguration('search_strategy'),
            'cmd_vel_topic': '/model/three_wheel_robot/cmd_vel',
            'odom_topic': '/model/three_wheel_robot/odometry',
        }]
    )
    
    return LaunchDescription([
        initial_city_arg,
        goal_city_arg,
        strategy_arg,
        bridge,
        navigation_node,
    ])
