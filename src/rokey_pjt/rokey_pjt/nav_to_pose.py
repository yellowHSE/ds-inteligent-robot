#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from geometry_msgs.msg import PoseStamped, Pose2D
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from tf_transformations import quaternion_from_euler

import time


def pose2d_to_posestamped(p2d: Pose2D, navigator: BasicNavigator) -> PoseStamped:
    """
    Pose2D(x[m], y[m], theta[rad]) → PoseStamped('map') 변환
    - Nav2는 PoseStamped를 사용하므로 변환이 필요
    """
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = float(p2d.x)
    pose.pose.position.y = float(p2d.y)

    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(p2d.theta))
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def create_pose(x, y, yaw_deg, navigator):
    """x, y, yaw(도 단위) → PoseStamped 생성 (초기 자세 설정용)"""
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)

    yaw_rad = float(yaw_deg) * 3.141592653589793 / 180.0
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw_rad)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


class FreeSlotNavigator(Node):
    """
    /parking/free_slot (Pose2D) 수신 → Nav2로 이동 후 도킹까지 처리하는 단일 노드
    - 이 노드는 자체 rclpy.Node이며, 내부에 BasicNavigator/TurtleBot4Navigator를 함께 사용
    - free_slot을 최초 1회만 처리
    """

    def __init__(self):
        super().__init__('nav_to_free_slot')

        # 두 navigator 인스턴스 생성 (네임스페이스는 프로세스 인자 __ns로 적용됨)
        self.dock_navigator = TurtleBot4Navigator()
        self.nav_navigator = BasicNavigator(node_name='navigator_robot4')

        # 초기 위치 설정 (필요 시 값 조정)
        initial_pose = create_pose(-0.01, -0.01, 0.0, self.nav_navigator)  # NORTH
        self.nav_navigator.setInitialPose(initial_pose)
        self.nav_navigator.get_logger().info('초기 위치 설정 중...')
        time.sleep(1.0)  # AMCL이 초기 pose 처리 및 TF ready 대기
        self.nav_navigator.waitUntilNav2Active()

        # 도킹되어 있다면 언도킹
        if self.dock_navigator.getDockedStatus():
            self.dock_navigator.get_logger().info('현재 도킹 상태 → 언도킹 시도')
            self.dock_navigator.undock()
        else:
            self.dock_navigator.get_logger().info('언도킹 상태에서 시작함')

        # /parking/free_slot 구독 (절대 토픽명: 전역 슬롯 할당 수신)
        qos = QoSProfile(depth=10)
        self.goal_received = False
        self.subscription = self.create_subscription(
            Pose2D,
            '/parking/free_slot',
            self._on_free_slot,
            qos
        )
        self.get_logger().info('대기 중: /parking/free_slot (Pose2D) 수신하면 이동 시작')

        # 주기적으로 spin_once를 돌며 Nav2 진행상황 표출
        self.create_timer(0.2, self._spin_nav2_feedback)

    def _on_free_slot(self, msg: Pose2D):
        if self.goal_received:
            return
        self.goal_received = True

        goal_pose = pose2d_to_posestamped(msg, self.nav_navigator)
        self.nav_navigator.get_logger().info(
            f'free_slot 수신 → 목표 이동 시작: x={msg.x:.3f}, y={msg.y:.3f}, theta(rad)={msg.theta:.3f}'
        )
        self.nav_navigator.goToPose(goal_pose)

    def _spin_nav2_feedback(self):
        """Nav2 진행 상황을 주기적으로 확인하고, 완료 시 결과 처리"""
        if not self.goal_received:
            return

        if not self.nav_navigator.isTaskComplete():
            feedback = self.nav_navigator.getFeedback()
            if feedback:
                remaining = feedback.distance_remaining
                self.nav_navigator.get_logger().info(f'남은 거리: {remaining:.2f} m')
            return

        # 작업 완료 → 결과 처리 (한 번만)
        result = self.nav_navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.nav_navigator.get_logger().info('목표 위치 도달 성공 → 도킹 시도')
            # self.dock_navigator.dock()
            # self.dock_navigator.get_logger().info('도킹 요청 완료')
        elif result == TaskResult.CANCELED:
            self.nav_navigator.get_logger().warn('이동이 취소되었습니다.')
        elif result == TaskResult.FAILED:
            error_code, error_msg = self.nav_navigator.getTaskError()
            self.nav_navigator.get_logger().error(f'이동 실패: {error_code} - {error_msg}')
        else:
            self.nav_navigator.get_logger().warn('알 수 없는 결과 코드 수신')

        # 결과 처리 이후에는 더 이상 타이머에서 반복 로직 수행하지 않도록 플래그 초기화
        self.goal_received = False

    def destroy(self):
        # 생성한 navigator와 노드 정리
        try:
            self.dock_navigator.destroy_node()
        except Exception:
            pass
        try:
            self.nav_navigator.destroy_node()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = FreeSlotNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('종료 요청 수신')
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()