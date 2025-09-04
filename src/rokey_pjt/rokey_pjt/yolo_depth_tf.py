#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 + Depth 융합 + TF 변환(→ map)
- RGB 프레임에서 YOLO 추론 → bbox 중심 주변의 깊이(윈도우 중앙값)
- Depth 인코딩: uint16(mm)/float32(m) 자동 처리, 깊이 시각화 컬러맵 제공
- RGB/Depth 해상도 불일치 대응(스케일 매핑), message_filters로 시간 동기화
- CameraInfo(K) 수신 시 3D(X,Y,Z[m]) 계산
- **TF2로 카메라(depth optical) 좌표의 3D 포인트를 target_frame(기본 'map')으로 변환**
- 헤드리스 옵션, 주석 이미지 퍼블리시(/yolo_depth/annotated)

권장:
  * 가능하면 드라이버에서 "depth → RGB 정렬"(alignment)을 켜세요. 정렬된 경우 `assume_aligned=true` 권장.
  * OAK-D처럼 RGB/Depth FOV가 다른 장치에선 좌표 스케일만으론 완전 정합이 되지 않습니다.

의존:
  pip install ultralytics
  ROS2: message_filters, cv_bridge, sensor_msgs, tf2_ros
"""

import os
import math
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

# 동기화
from message_filters import Subscriber as MFSubscriber, ApproximateTimeSynchronizer

# TF2
import tf2_ros
# 중요: geometry_msgs PointStamped 변환 지원 등록/함수 사용
from tf2_geometry_msgs import do_transform_point

# YOLOv8 (ultralytics)
from ultralytics import YOLO

WINDOW_NAME_RGB = 'YOLO + Depth + Map'
WINDOW_NAME_DEPTH = 'Depth Visualization'

class YoloDepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth')

        # ===== 공용 유틸 =====
        self.bridge = CvBridge()
        self.K = None               # CameraInfo(K) — depth 카메라 내부행렬
        self.fx = self.fy = self.cx = self.cy = None

        self.depth_cv = None        # 최신 depth
        self.depth_colored = None

        # ===== 파라미터 =====
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(os.path.dirname(pkg_dir), 'models', 'yolov8n.pt')

        # 토픽/모델/추론
        self.declare_parameter('rgb_topic', '/robot3/oakd/rgb/image_raw')
        self.declare_parameter('depth_topic', '/robot3/oakd/stereo/image_raw')
        self.declare_parameter('camera_info_topic', '/robot3/oakd/stereo/camera_info')
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('iou_thres', 0.5)
        self.declare_parameter('imgsz', 640)

        # 깊이/시각화
        self.declare_parameter('depth_window', 5)
        self.declare_parameter('normalize_depth_range', 3.0)
        self.declare_parameter('enable_viewer', True)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('annotated_topic', '/yolo_depth/annotated')
        self.declare_parameter('log_with_get_logger', True)

        # 좌표 출력
        self.declare_parameter('show_3d', True)          # 카메라(depth optical) 좌표 XYZ 라벨
        self.declare_parameter('show_map', True)         # map 좌표 라벨
        self.declare_parameter('assume_aligned', False)  # RGB-Depth 정렬/동일해상도 가정

        # TF 변환
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('tf_timeout_sec', 0.5)
        self.declare_parameter('transform_start_delay_sec', 5.0)  # TF 트리 안정화 대기

        # 로드
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.iou_thres = float(self.get_parameter('iou_thres').value)
        self.imgsz = int(self.get_parameter('imgsz').value)

        self.depth_window = int(self.get_parameter('depth_window').value)
        if self.depth_window % 2 == 0:
            self.depth_window += 1
        self.normalize_depth_range = float(self.get_parameter('normalize_depth_range').value)
        self.enable_viewer = bool(self.get_parameter('enable_viewer').value)
        self.publish_annotated = bool(self.get_parameter('publish_annotated').value)
        self.annotated_topic = self.get_parameter('annotated_topic').value
        self.log_with_get_logger = bool(self.get_parameter('log_with_get_logger').value)
        self.show_3d = bool(self.get_parameter('show_3d').value)
        self.show_map = bool(self.get_parameter('show_map').value)
        self.assume_aligned = bool(self.get_parameter('assume_aligned').value)

        self.target_frame = self.get_parameter('target_frame').value
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.transform_start_delay_sec = float(self.get_parameter('transform_start_delay_sec').value)

        # ===== YOLO 모델 =====
        if not os.path.exists(self.model_path):
            self.get_logger().warn(f"YOLO 가중치가 보이지 않습니다: {self.model_path}")
        self.model = YOLO(self.model_path)
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to('cuda')
                self.get_logger().info('Using CUDA for YOLO inference')
            else:
                self.get_logger().info('Using CPU for YOLO inference')
        except Exception:
            self.get_logger().info('torch 확인 실패 → CPU 사용')
        self.get_logger().info(f"YOLO model loaded: {self.model_path}")

        # ===== 퍼블리셔 =====
        self.annotated_pub = None
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 10)

        # ===== 구독 + 동기화 =====
        self.rgb_sub = MFSubscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = MFSubscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.sync_callback)

        self.caminfo_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data
        )

        # ===== TF Buffer/Listener + 지연 시작 =====
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_ready = False
        self.get_logger().info(f"TF Tree 안정화 대기… {self.transform_start_delay_sec:.1f}s 후 변환 시작")
        self.create_timer(self.transform_start_delay_sec, self._enable_tf)

        # 뷰어 준비
        if self.enable_viewer:
            try:
                cv2.namedWindow(WINDOW_NAME_RGB, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME_RGB, 640, 480)
                cv2.namedWindow(WINDOW_NAME_DEPTH, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME_DEPTH, 640, 480)
            except Exception as e:
                self.get_logger().warn(f"OpenCV 창 생성 실패(헤드리스?): {e}")
                self.enable_viewer = False

        self.get_logger().info(
            f"Subscribed:  RGB: {self.rgb_topic}  DEPTH: {self.depth_topic}  CameraInfo: {self.camera_info_topic}"
        )

    # ===== 내부 이벤트 =====
    def _enable_tf(self):
        self.tf_ready = True
        self.get_logger().info("TF Tree 안정화 완료. 변환 시작합니다.")

    # ===== 콜백들 =====
    def camera_info_callback(self, msg: CameraInfo):
        if self.K is None:
            K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.K = K
            self.fx, self.fy = K[0, 0], K[1, 1]
            self.cx, self.cy = K[0, 2], K[1, 2]
            self.get_logger().info(
                f"CameraInfo: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f} (frame={msg.header.frame_id})"
            )

    def sync_callback(self, rgb_msg: Image, depth_msg: Image):
        # to OpenCV
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        self.depth_cv = np.array(depth_raw, copy=False)

        # Depth 시각화
        if self.enable_viewer:
            depth_vis = self._depth_to_vis(self.depth_cv, self.normalize_depth_range)
            try:
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                self.depth_colored = depth_color
                cv2.imshow(WINDOW_NAME_DEPTH, depth_color)
            except Exception:
                pass

        # YOLO 추론
        t0 = time.perf_counter()
        results = self.model(rgb, conf=self.conf_thres, iou=self.iou_thres, imgsz=self.imgsz, verbose=True)[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        h_rgb, w_rgb = rgb.shape[:2]
        h_d, w_d = self.depth_cv.shape[:2]
        scale_x = (w_d / w_rgb) if not self.assume_aligned else 1.0
        scale_y = (h_d / h_rgb) if not self.assume_aligned else 1.0

        out = rgb.copy()

        # 변환 기준 프레임(깊이 프레임을 기준으로 3D를 만들기 때문에 depth frame 사용 권장)
        depth_frame_id = depth_msg.header.frame_id

        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                label = self.model.names.get(cls_id, str(cls_id))

                # 중심점 (RGB 좌표계)
                cx_rgb = max(0, min((x1 + x2) // 2, w_rgb - 1))
                cy_rgb = max(0, min((y1 + y2) // 2, h_rgb - 1))

                # Depth 좌표로 스케일 매핑
                cx_d = int(round(cx_rgb * scale_x))
                cy_d = int(round(cy_rgb * scale_y))
                cx_d = max(0, min(cx_d, w_d - 1))
                cy_d = max(0, min(cy_d, h_d - 1))

                # 중심 주변 패치의 중앙값 (m)
                dist_m = self._get_depth_median_m(self.depth_cv, cx_d, cy_d, self.depth_window)

                # 카메라(depth optical) 좌표의 3D 포인트 계산 (K 필요)
                # 간단한 파란색 텍스트 라벨 (예: "Car 0.41 1.10m")
                label_text = None

                # map 좌표 로그 출력만 (라벨엔 cam/map 미표기)
                if self.K is not None and dist_m and not math.isnan(dist_m):
                    Z = dist_m
                    X = (cx_d - self.cx) * Z / self.fx
                    Y = (cy_d - self.cy) * Z / self.fy

                    if self.tf_ready:
                        try:
                            pt_cam = PointStamped()
                            pt_cam.header.stamp = depth_msg.header.stamp
                            pt_cam.header.frame_id = depth_frame_id
                            pt_cam.point.x = float(X)
                            pt_cam.point.y = float(Y)
                            pt_cam.point.z = float(Z)

                            tr = self.tf_buffer.lookup_transform(
                                self.target_frame, depth_frame_id,
                                depth_msg.header.stamp,
                                timeout=Duration(seconds=self.tf_timeout_sec)
                            )
                            pt_map = do_transform_point(pt_cam, tr)
                            self.get_logger().info(
                                f"[yolo_depth_to_map]: [TF] {label} -> {self.target_frame}: (x={pt_map.point.x:.2f}, y={pt_map.point.y:.2f}, z={pt_map.point.z:.2f})"
                            )
                        except Exception as e:
                            self.get_logger().warn(f"TF transform to {self.target_frame} failed: {e}")

                # 시각화
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(out, (cx_rgb, cy_rgb), 3, (0, 0, 255), -1)

                if dist_m is None or math.isnan(dist_m):
                    label_text = f"{label} {conf:.2f} ?m"
                    dist_str = "?m"
                else:
                    label_text = f"{label} {conf:.2f} {dist_m:.2f}m"
                    dist_str = f"{dist_m:.2f}m"

                # get_logger 로도 요약 출력
                if self.log_with_get_logger:
                    parts = [f"[det] {label}", f"conf={conf:.2f}", f"dist={dist_str}"]
                    try:
                        if 'X' in locals() and X is not None and 'Y' in locals() and 'Z' in locals() and Z is not None:
                            parts.append(f"cam=({X:.2f},{Y:.2f},{Z:.2f})")
                    except Exception:
                        pass
                    try:
                        if 'pt_map' in locals() and pt_map is not None:
                            parts.append(f"map=({pt_map.point.x:.2f},{pt_map.point.y:.2f},{pt_map.point.z:.2f})")
                    except Exception:
                        pass
                    self.get_logger().info(" ".join(parts))

                # self._draw_label_blue(out, label_text, x1, max(0, y1 - 8)))

        # FPS/시간 표시
        cv2.putText(out, f"infer: {infer_ms:.1f} ms", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # 퍼블리시/뷰
        if self.annotated_pub is not None:
            self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(out, encoding='bgr8'))
        if self.enable_viewer:
            try:
                cv2.imshow(WINDOW_NAME_RGB, out)
                cv2.waitKey(1)
            except Exception:
                pass

    # ===== 유틸 =====
    def _depth_to_vis(self, depth_np: np.ndarray, normalize_range_m: float) -> np.ndarray:
        if depth_np is None:
            return np.zeros((480, 640), dtype=np.uint8)
        depth_m = self._raw_depth_to_meters(depth_np)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, normalize_range_m)
        vis = (depth_m / normalize_range_m * 255.0).astype(np.uint8)
        return vis

    def _raw_depth_to_meters(self, depth_np: np.ndarray) -> np.ndarray:
        if depth_np is None:
            return None
        if np.issubdtype(depth_np.dtype, np.floating):
            return depth_np.astype(np.float32)
        else:
            return depth_np.astype(np.float32) / 1000.0  # mm → m

    def _get_depth_median_m(self, depth_np: np.ndarray, x: int, y: int, win: int) -> float:
        if depth_np is None:
            return None
        depth_m = self._raw_depth_to_meters(depth_np)
        if depth_m is None:
            return None
        h, w = depth_m.shape[:2]
        r = win // 2
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w, x + r + 1), min(h, y + r + 1)
        patch = depth_m[y0:y1, x0:x1].reshape(-1)
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    # def _draw_label_blue(self, img: np.ndarray, text: str, x: int, y: int):
    #     """블루 텍스트(배경 없음) 라벨"""
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     scale = 0.6
    #     thickness = 2
    #     y = max(16, y)
    #     cv2.putText(img, text, (x, y), font, scale, (255, 0, 0), thickness, cv2.LINE_AA), font, scale, (255, 255, 255), th, cv2.LINE_AA)


# ---- 내부 함수 버그 수정용 ----
def normalize_depth_m(normalize_range_m: float):
    # helper: 안전한 0~255 스케일 변환
    def _scale(depth_m: np.ndarray) -> np.ndarray:
        if normalize_range_m <= 0:
            return np.zeros_like(depth_m, dtype=np.uint8)
        return (np.clip(depth_m, 0.0, normalize_range_m) / normalize_range_m * 255.0).astype(np.uint8)
    return _scale


def main():
    rclpy.init()
    node = YoloDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
