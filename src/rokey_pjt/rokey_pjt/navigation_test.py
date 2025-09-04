#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # [ (x,y,mode) ] mode=1 전진, -1 후진
        self.waypoints = [
            (-0.831, -0.376, 1),
            (-1.004,  2.288, -1),
            (-0.551,  1.892, 1)
        ]
        self.current_wp = 0
        self.odom_pose = None
        self.start_pose = None
        self.tolerance = 0.10   # 10cm 도착 허용 오차
        self.speed = 0.20

        self.timer = self.create_timer(0.05, self.tick)

    def odom_cb(self, msg):
        self.odom_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def tick(self):
        if self.odom_pose is None: return
        if self.current_wp >= len(self.waypoints):
            self.pub.publish(Twist())  # 정지
            self.get_logger().info('All waypoints completed.')
            rclpy.shutdown()
            return

        target = self.waypoints[self.current_wp]
        distance = dist(self.odom_pose, (target[0], target[1]))

        if distance > self.tolerance:
            twist = Twist()
            twist.linear.x = self.speed * target[2]  # mode=1 전진, -1 후진
            self.pub.publish(twist)
        else:
            self.get_logger().info(f'Reached waypoint {self.current_wp+1}')
            self.current_wp += 1
            self.start_pose = self.odom_pose
            self.pub.publish(Twist())  # 정지

def main():
    rclpy.init()
    node = WaypointNavigator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()