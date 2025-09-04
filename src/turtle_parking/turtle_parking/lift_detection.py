#!/usr/bin/env python3
import os
import threading
import json
import numpy as np
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_srvs.srv import Trigger
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 (import side-effect for geometry_msgs conversions)

from ultralytics import YOLO

# =============== 설정 ===============
RGB_TOPIC = '/robot3/oakd/rgb/image_raw/compressed'
DEPTH_TOPIC = '/robot3/oakd/stereo/image_raw'
CAMERA_INFO_TOPIC = '/robot3/oakd/rgb/camera_info'
MODEL_PATH = "/home/rokey/runs/train/y8n_person_car/weights/best.pt"

TARGET_CLASS_ID = {0, 1}   # 0: person, 1: car
NAME_REMAP = {0: 'person', 1: 'car'}

MAX_DISTANCE = 10.0
MIN_DISTANCE = 0.05
DEPTH_SCALE = 0.82
DEPTH_OFFSET = -0.05
DISTANCE_ADD = 0.15

TIMER_PERIOD = 0.1  # 10 Hz
CAR_SAMPLES_N = 10  # 중앙값 계산용 샘플 개수


class LiftDetectionNode(Node):
    def __init__(self):
        super().__init__('lift_detection_node')

        # YOLO
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(MODEL_PATH)
            self.model = YOLO(MODEL_PATH)
            self.get_logger().info(f'Loaded YOLO model: {MODEL_PATH}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            self.model = None

        self.bridge = CvBridge()
        self._img_lock = threading.Lock()

        # 이미지 버퍼
        self.rgb_img = None
        self.depth_img = None
        self.rgb_stamp = None
        self.depth_stamp = None
        self.depth_frame_id = None

        # CameraInfo
        self.fx = self.fy = None
        self.cx = self.cy = None
        self.camera_info_frame = None

        # 구독 핸들(이미지 계열은 지연 생성/해제)
        self.rgb_sub = None
        self.depth_sub = None
        self.cinfo_sub = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
               # noqa: E121 (hanging indent)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 서비스 (시작/정지/리셋)
        self.srv_start = self.create_service(Trigger, '/parking_in/start_perception', self.start_cb)
        self.srv_stop  = self.create_service(Trigger, '/parking_in/lift_stop', self.lift_stop_cb)
        self.srv_reset = self.create_service(Trigger, '/parking_in/reset_perception', self.reset_cb)

        # 퍼블리셔
        self.map_pub = self.create_publisher(String, '/parking_in/objects_in_map', 10)            # human 목록
        self.final_car_pub = self.create_publisher(PoseStamped, '/parking_in/final_car_pose', 1)  # 확정 car "front" 위치

        # 타이머
        self.timer = self.create_timer(TIMER_PERIOD, self.process_timer_callback)

        # 상태
        self._busy = False
        self._last_proc_stamp = None
        self._active = False  # /parking_in/start_perception 호출 시 True

        # 차량 위치 샘플/락
        self.car_locked = False
        self.car_samples = []   # (x,y,z) in map (front)
        self.car_final_pose = None

        self.get_logger().info('LiftDetectionNode idle. Call /parking_in/start_perception to begin.')

    # ---------- 시작/정지/리셋 서비스 ----------
    def start_cb(self, req, res):
        try:
            if not self._active:
                self._active = True
                self._ensure_image_subscriptions()
                self.get_logger().info('[START] Activated by /parking_in/start_perception')
            res.success = True
            res.message = 'Perception started.'
        except Exception as e:
            res.success = False
            res.message = f'Failed to start: {e}'
        return res

    def lift_stop_cb(self, req, res):
        try:
            # 구독 해제
            if self.rgb_sub is not None:
                self.destroy_subscription(self.rgb_sub); self.rgb_sub = None
            if self.depth_sub is not None:
                self.destroy_subscription(self.depth_sub); self.depth_sub = None
            if self.cinfo_sub is not None:
                self.destroy_subscription(self.cinfo_sub); self.cinfo_sub = None

            # 버퍼/상태 초기화
            with self._img_lock:
                self.rgb_img = None
                self.depth_img = None
                self.rgb_stamp = None
                self.depth_stamp = None
                self.depth_frame_id = None
                self.fx = self.fy = None
                self.cx = self.cy = None
                self.camera_info_frame = None

            self._last_proc_stamp = None
            self._busy = False
            self._active = False

            # 차량 샘플 상태도 초기화(다음 사이클을 위해)
            self.car_locked = False
            self.car_samples = []
            self.car_final_pose = None

            self.get_logger().info('[STOP] Unsubscribed and returned to idle.')
            res.success = True
            res.message = 'Perception stopped & cleared.'
        except Exception as e:
            res.success = False
            res.message = f'Failed to stop: {e}'
        return res

    def reset_cb(self, req, res):
        # 차량 샘플 상태만 초기화
        self.car_locked = False
        self.car_samples = []
        self.car_final_pose = None
        self._last_proc_stamp = None
        self.get_logger().info('[RESET] Perception state reset (car samples cleared).')
        res.success = True
        res.message = 'Perception state reset.'
        return res

    # ---------- 구독 생성 ----------
    def _ensure_image_subscriptions(self):
        if self.rgb_sub is None:
            self.rgb_sub = self.create_subscription(CompressedImage, RGB_TOPIC, self.rgb_cb, 10)
        if self.depth_sub is None:
            self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, 10)
        if self.cinfo_sub is None:
            self.cinfo_sub = self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.cinfo_cb, 10)

    # ---------- 콜백 ----------
    def rgb_cb(self, msg: CompressedImage):
        try:
            rgb = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._img_lock:
                self.rgb_img = rgb
                self.rgb_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'[RGB] decode error: {e}')

    def depth_cb(self, msg: Image):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth_raw.dtype == np.uint16:
                depth_m = depth_raw.astype(np.float32) / 1000.0
            elif depth_raw.dtype == np.float32:
                depth_m = depth_raw
            else:
                depth_m = depth_raw.astype(np.float32)
            with self._img_lock:
                self.depth_img = depth_m
                self.depth_stamp = msg.header.stamp
                self.depth_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f'[Depth] decode error: {e}')

    def cinfo_cb(self, msg: CameraInfo):
        try:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_info_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f'CameraInfo parse failed: {e}')

    # ---------- 메인 처리 ----------
    def process_timer_callback(self):
        if not self._active or self._busy:
            return
        self._busy = True
        try:
            with self._img_lock:
                rgb = None if self.rgb_img is None else self.rgb_img.copy()
                depth = None if self.depth_img is None else self.depth_img.copy()
                rgb_stamp = self.rgb_stamp
                depth_frame = self.depth_frame_id
                depth_stamp = self.depth_stamp

            if rgb is None or depth is None or self.model is None:
                return

            # 같은 프레임 중복 처리 방지
            if self._last_proc_stamp == rgb_stamp:
                return
            self._last_proc_stamp = rgb_stamp

            h_rgb, w_rgb = rgb.shape[:2]
            results = self.model(rgb, verbose=False)

            out = {"car": [], "human": []}
            car_candidates_map = []   # 잠금 전 수집
            human_list_map = []

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None or len(r.boxes) == 0:
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls not in TARGET_CLASS_ID:
                        continue
                    label = NAME_REMAP.get(cls, f'class_{cls}')
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 사람은 항상, 차는 잠금 전만 처리
                    if label == 'car' and self.car_locked:
                        continue

                    d = self.depth_from_bbox(depth, x1, y1, x2, y2, w_rgb, h_rgb)
                    if np.isnan(d):
                        continue

                    # 3D (카메라/depth 프레임)
                    u = (x1 + x2) * 0.5
                    v = (y1 + y2) * 0.5
                    Xc, Yc, Zc = self.pixel_to_camera(u, v, d, w_rgb, h_rgb)

                    cam_frame = depth_frame if depth_frame else 'camera_link'
                    p_cam = PointStamped()
                    p_cam.header.stamp = depth_stamp if depth_stamp is not None else self.get_clock().now().to_msg()
                    p_cam.header.frame_id = cam_frame
                    p_cam.point.x, p_cam.point.y, p_cam.point.z = float(Xc), float(Yc), float(Zc)

                    # map → odom → base_link 폴백
                    target_frame, p_target = self.try_transform_chain(p_cam, ['map', 'odom', 'base_link'])
                    if p_target is None:
                        self.get_logger().warn('TF transform failed to all target frames.')
                        continue

                    if label == 'person':
                        human_list_map.append({
                            "x": float(p_target.point.x),
                            "y": float(p_target.point.y),
                            "z": float(p_target.point.z),
                        })
                    elif label == 'car':
                        # 여기에 얻는 점은 "프런트"를 의미한다고 정의
                        car_candidates_map.append((
                            float(p_target.point.x),
                            float(p_target.point.y),
                            float(p_target.point.z),
                        ))

            # 차 프런트 위치 샘플 확정(잠금 전)
            if not self.car_locked and len(car_candidates_map) > 0:
                c0 = car_candidates_map[0]  # 필요 시 최근접 후보 선택으로 변경 가능
                self.car_samples.append(c0)
                self.get_logger().info(f'[CAR SAMPLE] #{len(self.car_samples)}: {c0}')

                if len(self.car_samples) >= CAR_SAMPLES_N:
                    arr = np.array(self.car_samples)
                    med = np.median(arr, axis=0)
                    ps = PoseStamped()
                    ps.header.stamp = self.get_clock().now().to_msg()
                    ps.header.frame_id = 'map'
                    ps.pose.position.x = float(med[0])  # 프런트 x
                    ps.pose.position.y = float(med[1])  # 프런트 y
                    ps.pose.position.z = 0.0
                    ps.pose.orientation = Quaternion(w=1.0)  # yaw은 다른 서비스에서 제공
                    self.final_car_pub.publish(ps)
                    self.car_final_pose = ps
                    self.car_locked = True
                    self.get_logger().info(f'[CAR LOCKED] median=({med[0]:.3f}, {med[1]:.3f}) — published /parking_in/final_car_pose')

            # 사람 좌표 발행
            out["human"] = human_list_map
            out["car"] = []  # 요구대로 car 목록은 발행 안 함
            msg = String()
            msg.data = json.dumps([out], ensure_ascii=False)
            self.map_pub.publish(msg)

        finally:
            self._busy = False

    # ---------- 유틸 ----------
    def depth_from_bbox(self, depth_m: np.ndarray, x1, y1, x2, y2, w_rgb, h_rgb):
        if depth_m is None:
            return float('nan')
        h_d, w_d = depth_m.shape[:2]
        cx_rgb = (x1 + x2) / 2.0
        cy_rgb = (y1 + y2) / 2.0
        cx = int(cx_rgb * w_d / w_rgb)
        cy = int(cy_rgb * h_d / h_rgb)
        if not (0 <= cx < w_d and 0 <= cy < h_d):
            return float('nan')
        x0, x1c = max(0, cx - 1), min(w_d - 1, cx + 1)
        y0, y1c = max(0, cy - 1), min(h_d - 1, cy + 1)
        patch = depth_m[y0:y1c + 1, x0:x1c + 1]
        vals = patch.flatten()
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0.0]
        if vals.size == 0:
            return float('nan')
        d = float(np.median(vals))
        d = d * DEPTH_SCALE + DEPTH_OFFSET + DISTANCE_ADD
        return float(np.clip(d, MIN_DISTANCE, MAX_DISTANCE))

    def pixel_to_camera(self, u: float, v: float, Z: float, w: int, h: int) -> Tuple[float, float, float]:
        if self.fx and self.fy and self.cx is not None and self.cy is not None:
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy
            return X, Y, Z
        # 폴백: 정규화 좌표(-1..1) * Z
        xnorm = (u - (w / 2.0)) / (w / 2.0)
        ynorm = (v - (h / 2.0)) / (h / 2.0)
        X = xnorm * Z
        Y = ynorm * Z
        return X, Y, Z

    def try_transform_chain(self, p_in: PointStamped, targets) -> Tuple[Optional[str], Optional[PointStamped]]:
        for tgt in targets:
            try:
                p_out = self.tf_buffer.transform(p_in, tgt, timeout=Duration(seconds=0.5))
                return tgt, p_out
            except Exception:
                continue
        return None, None


def main(args=None):
    rclpy.init(args=args)
    node = LiftDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
