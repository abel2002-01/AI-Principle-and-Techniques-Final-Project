#!/usr/bin/env python3
"""
Launch file for the Ethiopia Cities World in Gazebo Sim (gz sim).

This launch file starts Gazebo Sim with the Ethiopia cities world.
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable


def generate_launch_description():
    # Package directory
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # World file
    world_file = os.path.join(pkg_dir, 'worlds', 'ethiopia_cities.world')
    
    # Start Gazebo Sim
    gazebo = ExecuteProcess(
        cmd=[
            'gz', 'sim',
            '-r',
            world_file
        ],
        output='screen'
    )

    return LaunchDescription([
        # Ensure Gazebo Sim can resolve assets relative to the world file
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=os.path.join(pkg_dir, 'worlds')
        ),
        gazebo,
    ])
