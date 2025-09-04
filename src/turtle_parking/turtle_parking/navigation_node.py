#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from nav2_msgs.action import BackUp
from rclpy.action import ActionClient
from tf_transformations import quaternion_from_euler
from paho.mqtt import client as mqtt_client
import queue
import json
import random
import time
from std_srvs.srv import Trigger
from parking_msgs.srv import LiftYaw
from parking_msgs.srv import DetectPerson
from geometry_msgs.msg import Point
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import BatteryState

# -------- Firebase --------
import firebase_admin
from firebase_admin import credentials, db

# === 동작 파라미터 (필요시 조정) ===
PLATE_START_TIMEOUT = 5.0   # /parking_in/lift_pose_success 응답 대기 최대 시간 (sec)
PLATE_DONE_TIMEOUT  = 12.0  # /parking_in/car_number 완료 신호 대기 최대 시간 (sec)

# 네비게이션 이동 좌표
poses = {
    "drop_zone": {"x": -0.937, "y": -2.223, "z": 0.0, "w": 0.70710678},
    "A-2": {"x": -1.5345726563651985, "y": 1.64006715530895, "z": -0.9971459197603816, "w": 0.07549844174036058},
    "B-1": {"x": -2.532, "y": 0.398, "z": -0.682, "w": 0.731},
    "pickup_zone": {"x": 0.42, "y": -1.433, "z": 0.685, "w": 0.729},
    "dock_exit_waypoint": {"x": -1.6027213050173332, "y": -0.5941569261430202, "z": 0.9889817328984277, "w": 0.14803760330815635},
}

class ParkingNavigator(Node):
    def __init__(self):
        super().__init__('parking_navigator')
        self.namespace = self.get_namespace()  # e.g., "/robot3"
        self.robot_id = (self.namespace.strip('/') or 'robot3')
        self.get_logger().info(f"Robot ID: {self.robot_id}")

        # ----- Firebase 설정 파라미터 -----
        self.declare_parameter('firebase_db_url', 'https://ds-intelligent-robot-default-rtdb.asia-southeast1.firebasedatabase.app')
        self.declare_parameter('firebase_cred_path', '')  # service-account json 경로(옵션)
        self.fb_db_url = self.get_parameter('firebase_db_url').get_parameter_value().string_value
        self.fb_cred_path = self.get_parameter('firebase_cred_path').get_parameter_value().string_value
        self.fb_ref = None
        self._init_firebase()

        # 배터리 상태
        self.battery_percent = None
        self.declare_parameter('battery_topic', '/battery_state')
        battery_topic = self.get_parameter('battery_topic').get_parameter_value().string_value
        self.create_subscription(BatteryState, battery_topic, self._on_battery, 10)

        # Nav2 Navigator
        self.nav_navigator = BasicNavigator(namespace=self.namespace)
        self.get_logger().info("Nav2 초기화 완료")

        # 도킹 Navigator
        self.dock_navigator = TurtleBot4Navigator(namespace=self.namespace)

        # 초기 위치 설정
        initial_pose = self.create_pose(0.100, -0.491, 0.096)
        self.nav_navigator.setInitialPose(initial_pose)
        self.get_logger().info("Nav2 초기 위치 설정 중...")
        time.sleep(1.0)
        self.nav_navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 준비 완료")

        # MQTT 설정
        self.broker = 't7d10172.ala.us-east-1.emqxsl.com'
        self.port = 8883
        self.username = 'rokey'
        self.password = '1234'
        self.topic3 = "robot3"
        self.client_id = f'python-mqtt-{random.randint(0,100)}'
        self.client = self.connect_mqtt()
        self.client.loop_start()  # MQTT 수신 스레드 시작

        # 큐로 MQTT 메시지 처리
        self.msg_queue = queue.Queue()

        # Trigger 서비스 (번호판 인식 시작/완료 수신)
        self.cli_start = self.create_client(Trigger, '/parking_in/lift_pose_success')
        while not self.cli_start.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("번호판 인식 시작 서비스 대기중...")

        self.srv_done = self.create_service(Trigger, '/parking_in/car_number', self.number_callback)
        self.get_logger().info("서비스 준비 완료: lift_pose_success(클라이언트) + car_number(서버)")

        self.plate_done = False

        # BackUp 액션 클라이언트
        self.backup_client = ActionClient(self, BackUp, f'{self.namespace}/backup')

        # DetectPerson 서비스 클라이언트
        self.detect_person_cli = self.create_client(DetectPerson, 'parking_out/detect_person')
    # ---------------- Firebase ----------------
    def _init_firebase(self):
        try:
            if not firebase_admin._apps:
                if self.fb_cred_path:
                    cred = credentials.Certificate(self.fb_cred_path)
                    firebase_admin.initialize_app(cred, {'databaseURL': self.fb_db_url})
                else:
                    # 런타임 환경 자격(예: ADC) 사용
                    firebase_admin.initialize_app(options={'databaseURL': self.fb_db_url})
            self.fb_ref = db.reference(f'robot_status/{self.robot_id}')
            self.get_logger().info("Firebase RTDB 연결 완료")
        except Exception as e:
            self.get_logger().warn(f"Firebase 초기화 실패: {e}")
            self.fb_ref = None

    def _fb_set_status(self, active: bool):
        if not self.fb_ref:
            return
        try:
            self.fb_ref.update({'status': bool(active)})
        except Exception as e:
            self.get_logger().warn(f"Firebase status 업데이트 실패: {e}")

    def _fb_update_final(self, active: bool = True):
        if not self.fb_ref:
            return
        try:
            payload = {'status': bool(active)}
            if self.battery_percent is not None:
                payload['battery'] = int(self.battery_percent)
            self.fb_ref.update(payload)
        except Exception as e:
            self.get_logger().warn(f"Firebase 최종 업데이트 실패: {e}")

    # ---------------- 배터리 콜백 ----------------
    def _on_battery(self, msg: BatteryState):
        # percentage가 [0,1] 또는 [0,100]로 오는 경우 모두 대응
        pct = msg.percentage
        if pct is None:
            return
        if pct <= 1.2:     # 0~1 스케일
            pct = pct * 100.0
        self.battery_percent = max(0, min(100, int(round(pct))))

    # ---------------- MQTT 연결 ----------------
    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
                client.subscribe(self.topic3)
            else:
                print(f"Failed to connect, return code {rc}")
        client = mqtt_client.Client(client_id=self.client_id, protocol=mqtt_client.MQTTv311)
        client.tls_set()
        client.username_pw_set(self.username, self.password)
        client.on_connect = on_connect
        client.on_message = self.on_message
        client.connect(self.broker, self.port)
        return client

    def on_message(self, client, userdata, msg):
        self.msg_queue.put(msg)

    # ---------------- 메시지 처리 ----------------
    def process_queue(self):
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            try:
                data = json.loads(msg.payload.decode())
                zone_exit = data.get("parking_zone_exit")
                zone_entry = data.get('parking_zone_entry', None)
                yaw = data.get('dropzone', None)
                if zone_exit:
                    self.get_logger().info(f"{zone_exit} 출차 작업 시작")
                    self.exit(zone_exit)
                elif yaw and zone_entry:
                    self.get_logger().info(f"{zone_entry} 입차 작업 시작")
                    self.entry(zone_entry)
            except json.JSONDecodeError:
                self.get_logger().warn(f"Non-JSON message received: {msg.payload.decode()}")

    # ---------------- 출차 동작 ----------------
    def exit(self, zone_exit):
        # 작업 시작 → Firebase status
        self._fb_set_status(True)

        try:
            self.undock()
            self.undock()
            self.undock()

            self.move(zone_exit)

            self.back(0.5, speed=0.15)
            time.sleep(0.5)

            self.move("pickup_zone")

            self.detect_and_move_to_person()

            self.back(0.5, speed=0.15)

            self.move("dock_exit_waypoint")

            self.dock()
        finally:
            # 작업 종료 → Firebase battery & status
            self._fb_update_final(active=True)
    # ---------------- 입차 동작 ----------------
    def entry(self, zone_entry):
        # 작업 시작 → Firebase status
        self._fb_set_status(True)

        try:
            self.undock()
            self.move("drop_zone")
            print('드롭존 도착 완료')

            # 1) 번호판 인식 시작 트리거
            self.get_logger().info("번호판 인식 시작 요청 (/parking_in/lift_pose_success)")
            req = Trigger.Request()
            future = self.cli_start.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=PLATE_START_TIMEOUT)

            if not future.done():
                self.get_logger().warn("번호판 인식 시작 응답 없음(타임아웃)")
                return

            result = future.result()
            if not result or not result.success:
                self.get_logger().warn(f"번호판 인식 노드 시작 실패 또는 거부: {getattr(result,'message','')}")
                return

            self.get_logger().info(f"번호판 인식 노드 시작 성공: {result.message}")

            # 2) 완료 신호 대기
            if not self.wait_for_plate_done(timeout_sec=PLATE_DONE_TIMEOUT):
                self.get_logger().warn("번호판 인식 완료 신호 타임아웃")
                return

            self.get_logger().info("번호판 인식 완료 확인 → 다음 경로로 이동")

            # 3) 리프트 동작
            self._wait_for_lift_done(1.57)

            self.move(zone_entry)
        finally:
            # 작업 종료 → Firebase battery & status
            self._fb_update_final(active=True)

    # 사람 감지 서비스 요청 - 응답(좌표값) - 로봇 안전거리 확보 이동
    def detect_and_move_to_person(self, zone = "pickup_zone"):
        person_pose2d = None

        while person_pose2d is None:
            req = DetectPerson.Request()
            req.zone = zone
            future = self.detect_person_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            if resp and resp.success:
                person_pose2d = resp.person_position
                self.get_logger().info(f"사람 좌표 수신: x={person_pose2d.x:.2f}, y={person_pose2d.y:.2f}, theta={person_pose2d.theta:.2f}")
            else:
                self.get_logger().info("사람 좌표 미수신, 대기 중...")
                time.sleep(0.5)

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose.position.x = person_pose2d.x
        pose_stamped.pose.position.y = person_pose2d.y
        q = quaternion_from_euler(0, 0, person_pose2d.theta)
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]

        self.nav_navigator.goToPose(pose_stamped)
        while not self.nav_navigator.isTaskComplete():
            time.sleep(0.1)
        self.get_logger().info("사람 위치 도착 완료")

    # ------------------- 리프트 완료 토픽 콜백 -------------------
    def _wait_for_lift_done(self, target_yaw: float):
        self.cli_lift_yaw = self.create_client(LiftYaw, "/parking_in/lift_yaw")

        if not self.cli_lift_yaw.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("⚠️ /parking_in/lift_yaw 서비스 준비되지 않음")
            return
        self.get_logger().info("✅ /parking_in/lift_yaw 서비스 연결 완료")

        self.srv_lift_done = self.create_service(
            Trigger,
            "/parking_in/lift_done",
            self._srv_lift_done_callback
        )
        self.get_logger().info("🛰️ /parking_in/lift_done 서비스 서버 시작")

        req = LiftYaw.Request()
        req.yaw = target_yaw
        future = self.cli_lift_yaw.call_async(req)
        future.add_done_callback(self._on_lift_yaw_response)

    def _on_lift_yaw_response(self, future):
        try:
            result = future.result()
            if result is not None:
                if result.accepted:
                    self.get_logger().info(f"LiftYaw accepted: {result.message}")
                else:
                    self.get_logger().error(f"LiftYaw rejected: {result.message}")
            else:
                self.get_logger().error("❌ LiftYaw 응답이 None")
        except Exception as e:
            self.get_logger().error(f"LiftYaw 서비스 호출 중 오류 발생: {e}")

    def _srv_lift_done_callback(self, request, response):
        try:
            self.get_logger().info("✅ lift_done 서비스 호출 수신 → 후속 동작 시작")
            self.active_waypoint_number = 1
            self.state = "GO_WAYPOINTS"
            self._publish_state()
            self.get_logger().info("🏁 _go_waypoints 시작")
            response.success = True
            response.message = "웨이포인트 이동 시작"
        except Exception as e:
            self.get_logger().error(f"lift_done 처리 중 오류: {e}")
            response.success = False
            response.message = str(e)
        return response

    def number_callback(self, request, response):
        self.get_logger().info("번호판 인식 완료 트리거 수신 (/parking_in/car_number)")
        self.plate_done = True
        response.success = True
        response.message = "네비게이션이 완료 신호 수신함"
        return response

    def wait_for_plate_done(self, timeout_sec: float) -> bool:
        self.get_logger().info("번호판 인식 완료 대기 중...")
        t0 = time.time()
        while rclpy.ok() and not self.plate_done:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t0 > timeout_sec:
                return False
        self.plate_done = False
        return True

    # ---------------- 도킹/언도킹 ----------------
    def undock(self):
        self.back(0.5)

    def dock(self):
        self.dock_navigator.dock()

    # ---------------- 이동 ----------------
    def move(self, zone):
        location = poses.get(zone)
        if location is None:
            self.get_logger().error(f"{zone} 정보 없음")
            return
        pose = self.create_pose_from_dict(location)
        self.nav_navigator.goToPose(pose)
        while not self.nav_navigator.isTaskComplete():
            time.sleep(0.1)
        self.get_logger().info(f"{zone} 도착 완료")
        time.sleep(0.5)

    # ---------------- 후진 (BackUp 액션) ----------------
    def back(self, distance_m: float, speed: float = 0.1, margin_sec: float = 3.0) -> bool:
        if not self.backup_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("BackUp 액션 서버 연결 실패")
            return False

        goal = BackUp.Goal()
        goal.target = Point()
        goal.target.x = float(distance_m)
        goal.target.y = 0.0
        goal.target.z = 0.0
        goal.speed    = float(abs(speed))

        expected = abs(distance_m) / max(abs(speed), 1e-6)
        goal.time_allowance = Duration(sec=int(expected + margin_sec))

        send_goal_future = self.backup_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn("BackUp 목표 거부됨")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        status = getattr(result, "status", None)

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f'후진 완료 (요청 {distance_m} m, 속도 {speed} m/s, allowance {int(expected+margin_sec)} s)'
            )
            time.sleep(0.1)
            return True
        else:
            self.get_logger().warn(f'후진 실패/취소 status={status} (요청 {distance_m} m)')
            return False

    # ---------------- Pose 생성 ----------------
    def create_pose(self, x, y, theta):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        q = quaternion_from_euler(0, 0, theta)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def create_pose_from_dict(self, d):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = d["x"]
        pose.pose.position.y = d["y"]
        pose.pose.orientation.x = d.get("x_ori", 0.0)
        pose.pose.orientation.y = d.get("y_ori", 0.0)
        pose.pose.orientation.z = d["z"]
        pose.pose.orientation.w = d["w"]
        return pose

    def _publish_state(self):
        pass

# ---------------- 메인 ----------------
def main(args=None):
    rclpy.init(args=args)
    node = ParkingNavigator()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.process_queue()
    except KeyboardInterrupt:
        pass
    finally:
        node.client.loop_stop()
        node.client.disconnect()
        rclpy.shutdown()

if __name__ == "__main__":
    main()