# import re
# import rclpy
# from rclpy.node import Node
# import firebase_admin
# from firebase_admin import credentials, db
# from pathlib import Path
# from parking_msgs.srv import Entry, Exit
# from std_msgs.msg import Bool
# import json
# import random
# from paho.mqtt import client as mqtt_client
# from collections import deque
# import time
# import threading

# def slot_sort_key(slot_id: str):
#     m = re.match(r'([A-Za-z]+)-(\d+)', slot_id)
#     if m:
#         return (m.group(1), int(m.group(2)))
#     return (slot_id, 0)

# class RobotDecision(Node):
#     def __init__(self):
#         super().__init__('decision_node')

#         # MQTT 설정
#         self.broker = 't7d10172.ala.us-east-1.emqxsl.com'
#         self.port = 8883
#         self.username = 'rokey'
#         self.password = '1234'
#         self.topic0 = "robot0"
#         self.topic3 = "robot3"
#         self.client_id = f'python-mqtt-{random.randint(0, 100)}'

#         # MQTT 연결
#         self.client = self.connect_mqtt()
#         self.client.loop_start()

#         self.robot0_state = None
#         self.robot3_state = None
#         self.robot0_battery = None
#         self.robot3_battery = None
#         self.yaw = None
#         self.exit = None

#         # 요청 큐
#         self.request_queue = deque()

#         # Firebase 초기화
#         pkg_root = Path(__file__).resolve().parent
#         cred_path = str(pkg_root / "keys" / "service-account.json")
#         db_url = "https://ds-intelligent-robot-default-rtdb.asia-southeast1.firebasedatabase.app"
#         cred = credentials.Certificate(cred_path)
#         if not firebase_admin._apps:
#             firebase_admin.initialize_app(cred, {"databaseURL": db_url})
        
#         self.entry_srv = self.create_service(
#             Entry,         # 서비스 타입
#             '/entry',       # 서비스 이름
#             self.entry_callback
#         )

#         self.exit_srv = self.create_service(
#             Exit,         # 서비스 타입
#             '/exit',       # 서비스 이름
#             self.exit_callback
#         )

#     def connect_mqtt(self) -> mqtt_client.Client:
#         def on_connect(client, userdata, flags, rc):
#             if rc == 0:
#                 print("Connected to MQTT Broker!")
#             else:
#                 print("Failed to connect, return code %d\n", rc)

#         client = mqtt_client.Client(client_id=self.client_id, protocol=mqtt_client.MQTTv311)
#         client.tls_set()
#         client.username_pw_set(self.username, self.password)
#         client.on_connect = on_connect
#         client.connect(self.broker, self.port)
#         return client

#     def entry_callback(self, request: Entry.Request, response: Entry.Response):
#         """서비스 요청 처리"""
#         self.yaw = request.angle
#         if self.yaw:
#             print("차가 들어왔습니다")
#             self.request_queue.append("entry")
#             print('요청 목록: ', self.request_queue)
#             response.success = True
#             # self.read_robots()
#             # self.create_timer(0.1, self.read_robots)
#             threading.Thread(target=self.read_robots, daemon=True).start()

#         else:
#             response.success = False
#         return response

#     def exit_callback(self, request: Exit.Request, response: Exit.Response):
#         """서비스 요청 처리"""
#         self.zone = request.zone
#         if self.zone:
#             print("출차 요청이 들어왔습니다")
#             self.request_queue.append("exit")
#             print('요청 목록: ', self.request_queue)
#             response.success = True
#             # self.read_robots()
#             # self.create_timer(0.1, self.read_robots)
#             threading.Thread(target=self.read_robots, daemon=True).start()
#         else:
#             response.success = False
#         return response

#     def read_robots(self):
#         try:
#             ref_robot = db.reference('/robot_status')
#             robots = ref_robot.get()
#             ref_records = db.reference('/records')
#             slot = ref_records.get()

#             if not robots:
#                 return
            
#             if not slot:
#                 return
            
#             if slot:
#                 latest_key = sorted(slot.keys())[-1]   # 마지막 recN
#                 latest_slot = slot[latest_key]
#                 self.slot_id = latest_slot.get("slot_id")
#                 # print("가장 최근 record slot_id:", self.slot_id)

            

#             self.robot0_state = robots["robot0"].get("status", None)
#             self.robot0_battery = robots["robot0"].get("battery", None)
#             self.robot3_state = robots["robot3"].get("status", None)
#             self.robot3_battery = robots["robot3"].get("battery", None)
#             print("로봇0 상태: ", self.robot0_state, ", 로봇3 상태: ", self.robot3_state)

#             self.process_queue()   # 큐 처리

#         except Exception as e:
#             self.get_logger().error(f"Failed to read robot status: {e}")

#     def process_queue(self):
#         if not self.request_queue:
#             return

#         # 큐 첫 번째 요청만 처리
#         handled = self.decision()   # decision 내부에서 self.request_queue[0] 사용
#         if handled:
#             self.request_queue.popleft()   # 처리 완료 → 큐 제거
#             # self.get_logger().info(f"{handled} 요청 처리 완료 → 큐에서 제거")
#             print('요청 목록: ', self.request_queue)
#             time.sleep(10)
#             # 다음 요청이 남아있으면 재귀적으로 처리
#             self.process_queue()

#     def decision(self):
#         if not self.request_queue:
#             return False

#         task = self.request_queue[0]  # 큐 맨 앞 요청 확인
#         handled = False

#         if not self.robot0_state and not self.robot3_state:
#             print('둘 다 작업중')
#             print('대기중인 로봇이 생길 때까지 기다립니다')
#             if not hasattr(self, 'timer') or self.timer is None:
#                 self.timer = self.create_timer(5.0, self.read_robots)
#             return False

#         else:
#             # 상태가 바뀌었으니 타이머 해제
#             if hasattr(self, 'timer') and self.timer is not None:
#                 self.timer.cancel()
#                 self.timer = None

#         # 로봇 상태 기반으로 publish 처리
#         if not self.robot0_state and self.robot3_state:
#             if task == "entry":
#                 print('로봇0 : 작업중, 로봇3 : 대기중')
#                 print('yaw 값을 전송합니다: ', self.yaw)
#                 print('주차 구역을 전송합니다: ', self.slot_id)
#                 self.publish_robot3({
#                     "dropzone": self.yaw,
#                     "parking_zone_entry": self.slot_id})
#                 handled = True
#             elif task == "exit":
#                 print('로봇0 : 작업중, 로봇3 : 대기중')
#                 self.publish_robot3({"parking_zone_exit": self.zone})
#                 handled = True

#         elif self.robot0_state and not self.robot3_state:
#             if task == "entry":
#                 print('로봇0 : 대기중, 로봇3 : 작업중')
#                 print('yaw 값을 전송합니다: ', self.yaw)
#                 print('주차 구역을 전송합니다: ', self.slot_id)
#                 self.publish_robot0({
#                     "dropzone": self.yaw,
#                     "parking_zone_entry": self.slot_id})
#                 handled = True
#             elif task == "exit":
#                 print('로봇0 : 대기중, 로봇3 : 작업중')
#                 self.publish_robot0({"parking_zone_exit": self.zone})
#                 handled = True

#         else:  # 둘 다 대기중
#             print('둘 다 대기중')
#             print("로봇0 배터리: ", self.robot0_battery, ", 로봇3 배터리: ", self.robot3_battery)

#             if self.robot0_battery >= self.robot3_battery:
#                 print('로봇0의 배터리가 더 많습니다')
#                 if task == "entry":
#                     print('yaw 값을 전송합니다: ', self.yaw)
#                     print('주차 구역을 전송합니다: ', self.slot_id)
#                     self.publish_robot3({
#                         "dropzone": self.yaw,
#                         "parking_zone_entry": self.slot_id})
#                     handled = True
#                 elif task == "exit":
#                     self.publish_robot0({"parking_zone_exit": self.zone})
#                     handled = True
#             else:
#                 print('로봇3의 배터리가 더 많습니다')
#                 if task == "entry":
#                     print('yaw 값을 전송합니다: ', self.yaw)
#                     print('주차 구역을 전송합니다: ', self.slot_id)
#                     self.publish_robot3({
#                         "dropzone": self.yaw,
#                         "parking_zone_entry": self.slot_id})
#                     handled = True
#                 elif task == "exit":
#                     self.publish_robot3({"parking_zone_exit": self.zone})
#                     handled = True

#         return handled


#     def publish_robot0(self, message: dict):
#         msg_json = json.dumps(message, ensure_ascii=False)
#         result = self.client.publish(self.topic0, msg_json, qos=1, retain=False)
#         status = result[0]
#         if status == 0:
#             print(f"Sent `{msg_json}` to `{self.topic0}`")
#         else:
#             print(f":x: Failed to send message to `{self.topic0}`")

#     def publish_robot3(self, message: dict):
#         msg_json = json.dumps(message, ensure_ascii=False)
#         result = self.client.publish(self.topic3, msg_json, qos=1, retain=False)
#         status = result[0]
#         if status == 0:
#             print(f"Sent `{msg_json}` to `{self.topic3}`")
#         else:
#             print(f":x: Failed to send message to `{self.topic3}`")
    
# def main(args=None):
#     rclpy.init(args=args)
#     node = RobotDecision()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()

import re
import rclpy
from rclpy.node import Node
import firebase_admin
from firebase_admin import credentials, db
from pathlib import Path
from parking_msgs.srv import Entry, Exit
from std_msgs.msg import Bool
import json
import random
from paho.mqtt import client as mqtt_client
from collections import deque
import time
import threading
from typing import Optional


def slot_sort_key(slot_id: str):
    m = re.match(r'([A-Za-z]+)-(\d+)', slot_id)
    if m:
        return (m.group(1), int(m.group(2)))
    return (slot_id, 0)


class RobotDecision(Node):
    def __init__(self):
        super().__init__('decision_node')

        # MQTT 설정
        self.broker = 't7d10172.ala.us-east-1.emqxsl.com'
        self.port = 8883
        self.username = 'rokey'
        self.password = '1234'
        self.topic0 = "robot0"
        self.topic3 = "robot3"
        self.client_id = f'python-mqtt-{random.randint(0, 100)}'

        # MQTT 연결
        self.client = self.connect_mqtt()
        self.client.loop_start()

        # 상태 변수
        self.robot0_state: Optional[bool] = None
        self.robot3_state: Optional[bool] = None
        self.robot0_battery: Optional[float] = None
        self.robot3_battery: Optional[float] = None
        self.yaw: Optional[float] = None
        self.zone: Optional[str] = None
        self.slot_id: Optional[str] = None  # ★ 최신 slot_id 저장

        # 요청 큐
        self.request_queue = deque()

        # Firebase 초기화
        pkg_root = Path(__file__).resolve().parent
        cred_path = str(pkg_root / "keys" / "service-account.json")
        db_url = "https://ds-intelligent-robot-default-rtdb.asia-southeast1.firebasedatabase.app"
        cred = credentials.Certificate(cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"databaseURL": db_url})

        # 서비스
        self.entry_srv = self.create_service(
            Entry, '/entry', self.entry_callback
        )
        self.exit_srv = self.create_service(
            Exit, '/exit', self.exit_callback
        )

        self.timer = None  # 폴링 타이머 핸들

    # ---------------------------
    # Firebase: 최신 slot_id 읽기
    # ---------------------------
    def fetch_latest_slot_id(self) -> Optional[str]:
        """
        /records 에서 가장 최근 recN의 slot_id를 읽어 self.slot_id에 반영.
        실패/없음 시 None 반환.
        """
        try:
            ref_records = db.reference('/records')
            records = ref_records.get()
            if not records or not isinstance(records, dict):
                return None

            # rec 키를 정렬하여 가장 최근 키 선택
            latest_key = sorted(records.keys())[-1]
            latest = records.get(latest_key, {})
            slot_id = latest.get("slot_id")
            if isinstance(slot_id, str) and slot_id.strip():
                self.slot_id = slot_id
                return slot_id
            return None
        except Exception as e:
            self.get_logger().error(f"fetch_latest_slot_id() failed: {e}")
            return None

    # ---------------------------
    # MQTT
    # ---------------------------
    def connect_mqtt(self) -> mqtt_client.Client:
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)

        client = mqtt_client.Client(client_id=self.client_id, protocol=mqtt_client.MQTTv311)
        client.tls_set()
        client.username_pw_set(self.username, self.password)
        client.on_connect = on_connect
        client.connect(self.broker, self.port)
        return client

    # ---------------------------
    # 서비스 콜백
    # ---------------------------
    def entry_callback(self, request: Entry.Request, response: Entry.Response):
        self.yaw = request.angle
        if self.yaw is not None:
            print("차가 들어왔습니다")
            self.request_queue.append("entry")
            print('요청 목록: ', self.request_queue)
            response.success = True
            # 백그라운드에서 로봇/슬롯 상태 읽기 시작
            threading.Thread(target=self.read_robots, daemon=True).start()
        else:
            response.success = False
        return response

    def exit_callback(self, request: Exit.Request, response: Exit.Response):
        self.zone = request.zone
        if self.zone:
            print("출차 요청이 들어왔습니다")
            self.request_queue.append("exit")
            print('요청 목록: ', self.request_queue)
            response.success = True
            # 백그라운드에서 로봇/슬롯 상태 읽기 시작
            threading.Thread(target=self.read_robots, daemon=True).start()
        else:
            response.success = False
        return response

    # ---------------------------
    # RTDB 읽기 + 큐 처리
    # ---------------------------
    def read_robots(self):
        try:
            # ★ 먼저 최신 slot_id 동기화
            self.fetch_latest_slot_id()

            ref_robot = db.reference('/robot_status')
            robots = ref_robot.get()

            if not robots:
                return

            self.robot0_state = robots.get("robot0", {}).get("status", None)
            self.robot0_battery = robots.get("robot0", {}).get("battery", None)
            self.robot3_state = robots.get("robot3", {}).get("status", None)
            self.robot3_battery = robots.get("robot3", {}).get("battery", None)
            print("로봇0 상태: ", self.robot0_state, ", 로봇3 상태: ", self.robot3_state)

            self.process_queue()

        except Exception as e:
            self.get_logger().error(f"Failed to read robot status: {e}")

    def process_queue(self):
        if not self.request_queue:
            return

        handled = self.decision()
        if handled:
            self.request_queue.popleft()
            print('요청 목록: ', self.request_queue)
            time.sleep(10)
            self.process_queue()

    # ---------------------------
    # 디스패치 의사결정
    # ---------------------------
    def decision(self):
        if not self.request_queue:
            return False

        task = self.request_queue[0]
        handled = False

        # 두 로봇 모두 작업 중
        if not self.robot0_state and not self.robot3_state:
            print('둘 다 작업중')
            print('대기중인 로봇이 생길 때까지 기다립니다')
            if not hasattr(self, 'timer') or self.timer is None:
                self.timer = self.create_timer(5.0, self.read_robots)
            return False
        else:
            # 상태 바뀜 → 폴링 타이머 해제
            if hasattr(self, 'timer') and self.timer is not None:
                self.timer.cancel()
                self.timer = None

        # 안전: entry 전송 시 slot_id 필수 확보
        if task == "entry" and not self.slot_id:
            # 즉시 동기화 재시도
            self.fetch_latest_slot_id()
            if not self.slot_id:
                print("entry 요청이지만 slot_id가 아직 없음 → 잠시 후 재시도")
                if not hasattr(self, 'timer') or self.timer is None:
                    self.timer = self.create_timer(3.0, self.read_robots)
                return False

        # 로봇 상태 기반 분기
        if not self.robot0_state and self.robot3_state:
            # 로봇0 작업중, 로봇3 대기중
            if task == "entry":
                print('로봇0 : 작업중, 로봇3 : 대기중')
                print('yaw 값을 전송합니다: ', self.yaw)
                print('주차 구역을 전송합니다: ', self.slot_id)
                self.publish_robot3({
                    "dropzone": self.yaw,
                    "parking_zone_entry": self.slot_id
                })
                handled = True
            elif task == "exit":
                print('로봇0 : 작업중, 로봇3 : 대기중')
                self.publish_robot3({"parking_zone_exit": self.zone})
                handled = True

        elif self.robot0_state and not self.robot3_state:
            # 로봇0 대기중, 로봇3 작업중
            if task == "entry":
                print('로봇0 : 대기중, 로봇3 : 작업중')
                print('yaw 값을 전송합니다: ', self.yaw)
                print('주차 구역을 전송합니다: ', self.slot_id)
                self.publish_robot0({
                    "dropzone": self.yaw,
                    "parking_zone_entry": self.slot_id
                })
                handled = True
            elif task == "exit":
                print('로봇0 : 대기중, 로봇3 : 작업중')
                self.publish_robot0({"parking_zone_exit": self.zone})
                handled = True

        else:
            # 둘 다 대기중
            print('둘 다 대기중')
            print("로봇0 배터리: ", self.robot0_battery, ", 로봇3 배터리: ", self.robot3_battery)

            if (self.robot0_battery or 0) >= (self.robot3_battery or 0):
                print('로봇0의 배터리가 더 많거나 같음')
                if task == "entry":
                    print('yaw 값을 전송합니다: ', self.yaw)
                    print('주차 구역을 전송합니다: ', self.slot_id)
                    # 배터리 우위 로봇에게 전송하도록 정책 조정 가능
                    self.publish_robot0({
                        "dropzone": self.yaw,
                        "parking_zone_entry": self.slot_id
                    })
                    handled = True
                elif task == "exit":
                    self.publish_robot0({"parking_zone_exit": self.zone})
                    handled = True
            else:
                print('로봇3의 배터리가 더 많음')
                if task == "entry":
                    print('yaw 값을 전송합니다: ', self.yaw)
                    print('주차 구역을 전송합니다: ', self.slot_id)
                    self.publish_robot3({
                        "dropzone": self.yaw,
                        "parking_zone_entry": self.slot_id
                    })
                    handled = True
                elif task == "exit":
                    self.publish_robot3({"parking_zone_exit": self.zone})
                    handled = True

        return handled

    # ---------------------------
    # MQTT Publish
    # ---------------------------
    def publish_robot0(self, message: dict):
        msg_json = json.dumps(message, ensure_ascii=False)
        result = self.client.publish(self.topic0, msg_json, qos=1, retain=False)
        status = result[0]
        if status == 0:
            print(f"Sent `{msg_json}` to `{self.topic0}`")
        else:
            print(f":x: Failed to send message to `{self.topic0}`")

    def publish_robot3(self, message: dict):
        msg_json = json.dumps(message, ensure_ascii=False)
        result = self.client.publish(self.topic3, msg_json, qos=1, retain=False)
        status = result[0]
        if status == 0:
            print(f"Sent `{msg_json}` to `{self.topic3}`")
        else:
            print(f":x: Failed to send message to `{self.topic3}`")


def main(args=None):
    rclpy.init(args=args)
    node = RobotDecision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
