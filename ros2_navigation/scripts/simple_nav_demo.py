#!/usr/bin/env python3
"""
Simple Navigation Demo for Traveling Ethiopia

This script demonstrates the navigation using BFS path planning
and drives the robot through waypoints in Gazebo.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import sys
import os
import time

# Add parent directory for importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from traveling_ethiopia.graph import StateSpaceGraph
from traveling_ethiopia.search_strategies import SearchSolver


# City coordinates in Gazebo world
CITY_COORDINATES = {
    "Addis Ababa": (0.0, 0.0),
    "Ambo": (-11.0, 3.0),
    "Jimma": (-25.0, -15.0),
    "Wolaita Sodo": (-20.0, -30.0),
    "Arba Minch": (-15.0, -45.0),
    "Konso": (-5.0, -55.0),
    "Moyale": (5.0, -70.0),
}


def create_graph():
    """Create navigation graph."""
    graph = StateSpaceGraph()
    graph.add_edge("Addis Ababa", "Ambo")
    graph.add_edge("Ambo", "Jimma")
    graph.add_edge("Jimma", "Wolaita Sodo")
    graph.add_edge("Wolaita Sodo", "Arba Minch")
    graph.add_edge("Arba Minch", "Konso")
    graph.add_edge("Konso", "Moyale")
    return graph


class SimpleNavigator(Node):
    def __init__(self, path):
        super().__init__('simple_navigator')
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        
        self.path = path
        self.waypoints = [CITY_COORDINATES[city] for city in path if city in CITY_COORDINATES]
        self.current_idx = 1  # Skip first (starting position)
        
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        self.timer = self.create_timer(0.1, self.nav_loop)
        
        self.get_logger().info(f"Path: {' -> '.join(path)}")
        self.get_logger().info(f"Starting navigation with {len(self.waypoints)} waypoints")
    
    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
    
    def nav_loop(self):
        if self.current_idx >= len(self.waypoints):
            self.get_logger().info("🎉 ARRIVED AT MOYALE! Navigation complete!")
            self.stop()
            rclpy.shutdown()
            return
        
        tx, ty = self.waypoints[self.current_idx]
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 2.0:
            city = self.path[self.current_idx]
            self.get_logger().info(f"✓ Reached {city}! ({self.current_idx}/{len(self.waypoints)-1})")
            self.current_idx += 1
            return
        
        heading = math.atan2(dy, dx)
        err = heading - self.yaw
        while err > math.pi: err -= 2*math.pi
        while err < -math.pi: err += 2*math.pi
        
        cmd = Twist()
        if abs(err) > 0.2:
            cmd.linear.x = 0.5
            cmd.angular.z = 1.5 * (1 if err > 0 else -1)
        else:
            cmd.linear.x = min(3.0, dist * 0.3)
            cmd.angular.z = err * 0.5
        
        self.cmd_pub.publish(cmd)
    
    def stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)


def main():
    # Plan path using BFS
    print("\n" + "="*60)
    print("TRAVELING ETHIOPIA - GAZEBO NAVIGATION DEMO")
    print("="*60)
    
    graph = create_graph()
    solver = SearchSolver(graph, "Addis Ababa", "Moyale")
    path = solver.solve("bfs")
    
    print(f"\nBFS Path: {' -> '.join(path)}")
    print(f"Starting navigation in Gazebo...\n")
    
    rclpy.init()
    nav = SimpleNavigator(path)
    
    try:
        rclpy.spin(nav)
    except KeyboardInterrupt:
        nav.stop()
    finally:
        nav.destroy_node()


if __name__ == '__main__':
    main()

