#!/usr/bin/env python3
"""
Navigation Node for Traveling Ethiopia Robot

Question 5.3: ROS-based class that uses uninformed search strategy
to generate a path for the robot to travel from any given initial state
to the given goal state.

This node:
1. Uses BFS/DFS from the traveling_ethiopia package for path planning
2. Converts city waypoints to Gazebo coordinates
3. Commands the robot to follow the path
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
import math
import sys
import os

# Add parent directory to path for importing traveling_ethiopia package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from traveling_ethiopia.graph import StateSpaceGraph
from traveling_ethiopia.search_strategies import SearchSolver


class EthiopiaNavigationNode(Node):
    """
    ROS2 Navigation node for the Traveling Ethiopia robot.
    
    Uses uninformed search strategies (BFS/DFS) to plan paths
    between Ethiopian cities and navigates the robot through waypoints.
    """
    
    # City coordinates in Gazebo world (x, y)
    # Based on the ethiopia_cities.world file
    CITY_COORDINATES = {
        "Addis Ababa": (0.0, 0.0),
        "Ambo": (-11.0, 3.0),
        "Jimma": (-25.0, -15.0),
        "Wolaita Sodo": (-20.0, -30.0),
        "Arba Minch": (-15.0, -45.0),
        "Konso": (-5.0, -55.0),
        "Moyale": (5.0, -70.0),
        "Adama": (10.0, -2.0),
        "Dire Dawa": (35.0, 5.0),
        "Harar": (40.0, 8.0),
        "Debre Berhan": (5.0, 13.0),
        "Dessie": (8.0, 30.0),
        "Woldia": (5.0, 40.0),
        "Lalibela": (-5.0, 45.0),
        "Mekelle": (15.0, 55.0),
        "Bahir Dar": (-30.0, 35.0),
        "Debre Markos": (-20.0, 15.0),
        "Gondar": (-35.0, 50.0),
        "Axum": (-20.0, 70.0),
        "Bale": (15.0, -25.0),
        "Assela": (12.0, -10.0),
        "Nekemte": (-25.0, 5.0),
        "Bonga": (-30.0, -20.0),
    }
    
    def __init__(self):
        super().__init__('ethiopia_navigation')
        
        # Declare parameters
        self.declare_parameter('initial_city', 'Addis Ababa')
        self.declare_parameter('goal_city', 'Moyale')
        self.declare_parameter('search_strategy', 'bfs')
        # Gazebo Sim topics by default (ros_gz_bridge)
        self.declare_parameter('cmd_vel_topic', '/model/three_wheel_robot/cmd_vel')
        self.declare_parameter('odom_topic', '/model/three_wheel_robot/odometry')
        
        # Get parameters
        self.initial_city = self.get_parameter('initial_city').value
        self.goal_city = self.get_parameter('goal_city').value
        self.strategy = self.get_parameter('search_strategy').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/robot/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/robot/imu', self.imu_callback, 10)
        
        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.obstacle_detected = False
        
        # Navigation state
        self.path = []
        self.waypoints = []
        self.current_waypoint_index = 0
        self.navigation_active = False
        
        # Navigation parameters
        self.linear_speed = 2.0
        self.angular_speed = 1.5
        self.goal_tolerance = 2.0
        self.heading_tolerance = 0.15
        
        # Create the graph
        self.graph = self._create_simplified_graph()
        
        # Timer for navigation control loop
        self.timer = self.create_timer(0.1, self.navigation_loop)
        
        self.get_logger().info(f"Navigation Node initialized")
        self.get_logger().info(f"Initial city: {self.initial_city}")
        self.get_logger().info(f"Goal city: {self.goal_city}")
        self.get_logger().info(f"Search strategy: {self.strategy}")
        self.get_logger().info(f"cmd_vel topic: {self.cmd_vel_topic}")
        self.get_logger().info(f"odom topic: {self.odom_topic}")
        
        # Start navigation after a delay
        self.create_timer(2.0, self.start_navigation, callback_group=None)
    
    def _create_simplified_graph(self) -> StateSpaceGraph:
        """Create a simplified graph matching the Gazebo world cities."""
        graph = StateSpaceGraph()
        
        # Add edges based on direct connections
        graph.add_edge("Addis Ababa", "Ambo")
        graph.add_edge("Addis Ababa", "Adama")
        graph.add_edge("Addis Ababa", "Debre Berhan")
        graph.add_edge("Addis Ababa", "Debre Markos")
        
        graph.add_edge("Ambo", "Nekemte")
        graph.add_edge("Ambo", "Jimma")
        
        graph.add_edge("Jimma", "Bonga")
        graph.add_edge("Jimma", "Wolaita Sodo")
        
        graph.add_edge("Wolaita Sodo", "Arba Minch")
        
        graph.add_edge("Arba Minch", "Konso")
        
        graph.add_edge("Konso", "Moyale")
        
        graph.add_edge("Adama", "Dire Dawa")
        graph.add_edge("Adama", "Assela")
        
        graph.add_edge("Assela", "Bale")
        
        graph.add_edge("Dire Dawa", "Harar")
        
        graph.add_edge("Debre Berhan", "Dessie")
        
        graph.add_edge("Dessie", "Woldia")
        
        graph.add_edge("Woldia", "Lalibela")
        graph.add_edge("Woldia", "Mekelle")
        
        graph.add_edge("Mekelle", "Axum")
        
        graph.add_edge("Debre Markos", "Bahir Dar")
        
        graph.add_edge("Bahir Dar", "Gondar")
        
        graph.add_edge("Gondar", "Axum")
        
        return graph
    
    def start_navigation(self):
        """Plan the path and start navigation."""
        if self.navigation_active:
            return
        
        self.get_logger().info(f"Planning path from {self.initial_city} to {self.goal_city}...")
        
        # Check if cities exist
        if self.initial_city not in self.CITY_COORDINATES:
            self.get_logger().error(f"Unknown city: {self.initial_city}")
            return
        if self.goal_city not in self.CITY_COORDINATES:
            self.get_logger().error(f"Unknown city: {self.goal_city}")
            return
        
        # Use search solver to find path
        solver = SearchSolver(self.graph, self.initial_city, self.goal_city)
        self.path = solver.solve(strategy=self.strategy)
        
        if self.path is None:
            self.get_logger().error(f"No path found from {self.initial_city} to {self.goal_city}")
            return
        
        self.get_logger().info(f"Path found using {self.strategy.upper()}: {' -> '.join(self.path)}")
        
        # Convert path to waypoints
        self.waypoints = []
        for city in self.path:
            if city in self.CITY_COORDINATES:
                self.waypoints.append(self.CITY_COORDINATES[city])
            else:
                self.get_logger().warn(f"City {city} not in coordinate map, skipping")
        
        if len(self.waypoints) > 1:
            # Start from second waypoint (assuming robot starts at first)
            self.current_waypoint_index = 1
            self.navigation_active = True
            self.get_logger().info(f"Starting navigation with {len(self.waypoints)} waypoints")
        else:
            self.get_logger().info("Already at goal!")
    
    def odom_callback(self, msg: Odometry):
        """Update robot position from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def scan_callback(self, msg: LaserScan):
        """Process laser scan for obstacle detection."""
        # Check front 60 degrees for obstacles
        num_readings = len(msg.ranges)
        if num_readings == 0:
            return
        
        # Get front readings
        front_start = int(num_readings * 0.4)
        front_end = int(num_readings * 0.6)
        front_ranges = msg.ranges[front_start:front_end]
        
        # Filter out invalid readings
        valid_ranges = [r for r in front_ranges if msg.range_min < r < msg.range_max]
        
        if valid_ranges:
            min_distance = min(valid_ranges)
            self.obstacle_detected = min_distance < 1.5
        else:
            self.obstacle_detected = False
    
    def imu_callback(self, msg: Imu):
        """Process IMU data for orientation verification."""
        # Could be used for more accurate orientation
        pass
    
    def navigation_loop(self):
        """Main navigation control loop."""
        if not self.navigation_active:
            return
        
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info("Navigation complete! Reached destination.")
            self.stop_robot()
            self.navigation_active = False
            return
        
        target_x, target_y = self.waypoints[self.current_waypoint_index]
        
        # Calculate distance and heading to target
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.sqrt(dx*dx + dy*dy)
        target_heading = math.atan2(dy, dx)
        
        # Calculate heading error
        heading_error = target_heading - self.current_yaw
        # Normalize to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Check if reached current waypoint
        if distance < self.goal_tolerance:
            city_name = self.path[self.current_waypoint_index] if self.current_waypoint_index < len(self.path) else "Unknown"
            self.get_logger().info(f"Reached {city_name}! Distance: {distance:.2f}m")
            self.current_waypoint_index += 1
            return
        
        # Generate velocity command
        cmd = Twist()
        
        if self.obstacle_detected:
            # Obstacle avoidance - turn in place
            self.get_logger().warn("Obstacle detected! Avoiding...")
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed
        elif abs(heading_error) > self.heading_tolerance:
            # Turn towards target
            cmd.linear.x = 0.2
            cmd.angular.z = self.angular_speed * heading_error / abs(heading_error)
        else:
            # Move towards target
            cmd.linear.x = min(self.linear_speed, distance)
            cmd.angular.z = 0.5 * heading_error
        
        self.cmd_vel_pub.publish(cmd)
    
    def stop_robot(self):
        """Stop the robot."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    
    node = EthiopiaNavigationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Navigation interrupted")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
