# Question 5: ROS2 Navigation Robot for Traveling Ethiopia

This package implements a three-wheeled robot simulation in Gazebo Sim that can navigate
through Ethiopian cities using uninformed search strategies.

## Requirements

- ROS 2 Jazzy (or later)
- Gazebo Sim (`gz sim`)
- Python 3.10+
- ros_gz_bridge

## Installation

```bash
# Install ROS 2 Jazzy (if not installed)
# Then install Gazebo Sim and ros_gz_bridge (ROS-GZ bridge)
# (Exact package names vary by distro)
```

## Package Structure

```
ros2_navigation/
├── README.md
├── setup.py
├── package.xml
├── launch/
│   ├── robot_launch.py
│   └── world_launch.py
├── urdf/
│   └── three_wheel_robot.urdf.xacro
├── worlds/
│   └── ethiopia_cities.world
├── config/
│   └── navigation_params.yaml
└── scripts/
    └── navigation_node.py
```

## Usage

### Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select traveling_ethiopia_robot
source install/setup.bash
```

### Launch the simulation (recommended)

```bash
ros2 launch traveling_ethiopia_robot sim_launch.py
```

### Launch in two terminals (world + robot control)

```bash
# Terminal 1: world
ros2 launch traveling_ethiopia_robot world_launch.py

# Terminal 2: bridge + navigation
ros2 launch traveling_ethiopia_robot robot_launch.py \
  initial_city:="Addis Ababa" goal_city:="Moyale" search_strategy:="bfs"
```

If you run Gazebo Sim manually, set the resource path so textures load:

```bash
export GZ_SIM_RESOURCE_PATH=$(ros2 pkg prefix traveling_ethiopia_robot)/share/traveling_ethiopia_robot/worlds
gz sim -r $GZ_SIM_RESOURCE_PATH/ethiopia_cities.world
```

## Features

### 5.1 Three-Wheel Robot Design
- Differential drive robot with two drive wheels and one caster
- Working physics engine with appropriate mass and inertia
- Sensors included:
  - Proximity sensor (laser scanner)
  - Gyroscope (IMU)
  - RGB Camera
  
Note: The world file already contains the robot model converted for Gazebo Sim.

### 5.2 World File
- Ethiopian cities represented in Cartesian coordinates
- Landmarks visible for each city
- Navigable terrain between cities

### 5.3 Navigation
- Implements BFS/DFS for path planning
- Converts graph paths to waypoints
- Uses direct velocity control via ros_gz_bridge
