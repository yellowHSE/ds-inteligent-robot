#!/usr/bin/env python3
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
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped, Pose2D
from visualization_msgs.msg import Marker
# service
from parking_msgs.srv import DetectPerson
from message_filters import Subscriber as MFSubscriber, ApproximateTimeSynchronizer

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from ultralytics import YOLO

WINDOW_NAME_RGB = 'YOLO + Depth + Map'
WINDOW_NAME_DEPTH = 'Depth Visualization'

# 사람 앞에서 정지할 오프셋 (카메라→사람 방향으로 0.5 m 당김)
APPROACH_OFFSET_M = 0.5
MIN_Z_CLAMP = 0.05  # 너무 가까워져 음수/0 방지


class YoloDepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth')

        self.bridge = CvBridge()
        self.K = None
        self.fx = self.fy = self.cx = self.cy = None

        self.depth_cv = None
        self.depth_colored = None

        # -------- 서비스용 변수 초기화 --------
        self.last_person_pose_map = None

        # -------- Parameters --------
        self.declare_parameter('rgb_topic', '/robot0/oakd/rgb/image_raw/compressed')
        self.declare_parameter('depth_topic', '/robot0/oakd/stereo/image_raw')
        self.declare_parameter('camera_info_topic', '/robot0/oakd/stereo/camera_info')

        self.declare_parameter('model_path', '/home/rokey/rokey_ws/src/rokey_pjt/models/best.pt')
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('iou_thres', 0.5)
        self.declare_parameter('imgsz', 640)

        self.declare_parameter('depth_window', 5)
        self.declare_parameter('normalize_depth_range', 3.0)
        self.declare_parameter('enable_viewer', True)
        self.declare_parameter('publish_annotated', False)
        self.declare_parameter('annotated_topic', '/yolo_depth/annotated')
        self.declare_parameter('log_with_get_logger', True)

        self.declare_parameter('show_3d', True)
        self.declare_parameter('show_map', True)
        self.declare_parameter('assume_aligned', False)

        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('tf_timeout_sec', 0.5)
        self.declare_parameter('transform_start_delay_sec', 5.0)

        # --- marker topic param ---
        self.declare_parameter('marker_topic', '/robot0/person_markers')

        # -------- Load params --------
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

        self.marker_topic = self.get_parameter('marker_topic').value

        # -------- YOLO --------
        if not os.path.exists(self.model_path):
            self.get_logger().warn(f"YOLO weight not found: {self.model_path}")
        self.model = YOLO(self.model_path)
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to('cuda')
                self.get_logger().info('Using CUDA for YOLO inference')
            else:
                self.get_logger().info('Using CPU for YOLO inference')
        except Exception:
            self.get_logger().info('torch not available → CPU')

        # -------- Publishers --------
        self.annotated_pub = None
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 10)

        # Pose2D (x, y, theta) 퍼블리셔 — 오프셋 적용 결과를 발행
        self.pose_pub = self.create_publisher(Pose2D, '/robot0/person_in_map', 10)

        # RViz Marker publisher (참고 코드 스타일)
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, 10)
        self.marker_id = 0

        # -------- Subs + sync --------
        self.rgb_sub = MFSubscriber(self, CompressedImage, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = MFSubscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=20, slop=0.15)
        self.ts.registerCallback(self.sync_callback)

        self.caminfo_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data
        )

        # -------- TF --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_ready = False
        self.get_logger().info(f"Waiting TF stabilization {self.transform_start_delay_sec:.1f}s…")
        self.create_timer(self.transform_start_delay_sec, self._enable_tf)

        # -------- 서비스 --------
        self.detect_person_srv = self.create_service(
            DetectPerson, '/parking_out/detect_person', self.detect_person_callback
        )
        self.get_logger().info("Service '/parking_out/detect_person' ready.")

        # -------- Windows --------
        if self.enable_viewer:
            try:
                cv2.namedWindow(WINDOW_NAME_RGB, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME_RGB, 640, 480)
                cv2.namedWindow(WINDOW_NAME_DEPTH, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME_DEPTH, 640, 480)
            except Exception as e:
                self.get_logger().warn(f"OpenCV window creation failed: {e}")
                self.enable_viewer = False

        self.get_logger().info(
            f"Subscribed: RGB={self.rgb_topic}  DEPTH={self.depth_topic}  CameraInfo={self.camera_info_topic}  Marker={self.marker_topic}"
        )

    # ---------- helpers ----------
    def _enable_tf(self):
        self.tf_ready = True
        self.get_logger().info("TF ready. Start transforms.")

    def camera_info_callback(self, msg: CameraInfo):
        if self.K is None:
            K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.K = K
            self.fx, self.fy = K[0, 0], K[1, 1]
            self.cx, self.cy = K[0, 2], K[1, 2]
            self.get_logger().info(
                f"CameraInfo: fx={self.fx:.2f} fy={self.fy:.2f} cx={self.cx:.2f} cy={self.cy:.2f} frame={msg.header.frame_id}"
            )

    def _decode_depth(self, msg: Image):
        """Raw depth Image → numpy"""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            return depth
        except Exception as e:
            self.get_logger().warn(f"[Depth] decode failed: {e}")
            return None 
        
        # 2) 수동 복원
        fmt = getattr(msg, "format", "")
        data = memoryview(msg.data)
        HEADER_SZ = 12  # int32 + float32 + float32

        payload = data[HEADER_SZ:] if "Depth" in fmt else data

        # PNG 시그니처 검색
        PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
        bb = bytes(payload)
        i = bb.find(PNG_MAGIC)
        if i != -1:
            try:
                arr = np.frombuffer(bb[i:], np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    return img
            except Exception as e:
                self.get_logger().warn(f"[Depth] PNG decode error: {e}")

        # RVL(또는 비-PNG)
        self.get_logger().warn(
            "[Depth] payload looks like RVL or non-PNG. This node does not decode RVL. "
            "Please republish Depth as PNG or raw."
        )
        return None

    # ---------- 서비스 콜백 ----------
    def detect_person_callback(self, request, response):
        """
        request.zone: string (예: 'front', 'left', 'right')
        response.person_position: Pose2D
        response.success: bool
        """
        if self.last_person_pose_map is not None:
            response.person_position = self.last_person_pose_map
            response.success = True
        else:
            response.success = False
        return response

    # --- 참고 코드 스타일: 마커(SPHERE) 발행 함수 ---
    def publish_marker(self, x: float, y: float, z: float, label: str,
                       r: float = 1.0, g: float = 1.0, b: float = 0.0,
                       scale: float = 0.20, lifetime_sec: float = 5.0):
        """map 좌표에 구(SPHERE) 하나 찍기"""
        try:
            marker = Marker()
            marker.header.frame_id = self.target_frame  # 보통 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_objects'
            marker.id = self.marker_id
            self.marker_id += 1

            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)
            marker.pose.orientation.w = 1.0

            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            marker.color.r = float(r)
            marker.color.g = float(g)
            marker.color.b = float(b)
            marker.color.a = 1.0

            marker.lifetime = Duration(seconds=float(lifetime_sec)).to_msg()
            self.marker_pub.publish(marker)
        except Exception as e:
            self.get_logger().warn(f"[Marker] publish failed: {e}")

    def sync_callback(self, rgb_msg: CompressedImage, depth_msg: Image):
        # RGB decode
        rgb = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        # Depth decode (robust)
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth_raw is None:
            self.get_logger().warn(f"[Depth] decode failed (fmt='{getattr(depth_msg,'format','?')}', len={len(depth_msg.data)})")
            if self.enable_viewer:
                try:
                    cv2.imshow(WINDOW_NAME_RGB, rgb)
                    cv2.waitKey(1)
                except Exception:
                    pass
            return

        # 3채널로 오는 경우 첫 채널 사용
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]

        # mm → m
        if depth_raw.dtype == np.uint16:
            self.depth_cv = depth_raw.astype(np.float32) / 1000.0
        elif depth_raw.dtype == np.float32:
            self.depth_cv = depth_raw
        else:
            self.depth_cv = depth_raw.astype(np.float32)  # 이미 m 가정

        # depth view
        if self.enable_viewer:
            try:
                depth_vis = self._depth_to_vis(self.depth_cv, self.normalize_depth_range)
                self.depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow(WINDOW_NAME_DEPTH, self.depth_colored)
            except Exception:
                pass

        # YOLO
        t0 = time.perf_counter()
        results = self.model(rgb, conf=self.conf_thres, iou=self.iou_thres, imgsz=self.imgsz, verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        h_rgb, w_rgb = rgb.shape[:2]
        h_d, w_d = self.depth_cv.shape[:2]
        scale_x = (w_d / w_rgb) if not self.assume_aligned else 1.0
        scale_y = (h_d / h_rgb) if not self.assume_aligned else 1.0

        out = rgb.copy()
        depth_frame_id = depth_msg.header.frame_id

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().numpy())
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                label = self.model.names.get(cls_id, str(cls_id))

                cx_rgb = max(0, min((x1 + x2) // 2, w_rgb - 1))
                cy_rgb = max(0, min((y1 + y2) // 2, h_rgb - 1))

                cx_d = max(0, min(int(round(cx_rgb * scale_x)), w_d - 1))
                cy_d = max(0, min(int(round(cy_rgb * scale_y)), h_d - 1))

                # 중심 주변 패치의 중앙값 (m)
                Z = self._get_depth_median_m(self.depth_cv, cx_d, cy_d, self.depth_window)
                if Z is None or not np.isfinite(Z) or Z <= 0:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(out, f"{label} {conf:.2f} ?m", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    continue

                # 역투영 (카메라(depth) 좌표계)
                if self.K is not None:
                    X = (cx_d - self.cx) * Z / self.fx
                    Y = (cy_d - self.cy) * Z / self.fy
                else:
                    X = Y = None

                # ---- 오프셋 적용: 카메라→사람 방향으로 0.5 m 당김 ----
                if X is not None:
                    norm = math.sqrt(X*X + Y*Y + Z*Z)
                    if norm < 1e-6:
                        # 너무 가까움 — 스킵
                        continue
                    ux, uy, uz = X / norm, Y / norm, Z / norm
                    X_t = X - APPROACH_OFFSET_M * ux
                    Y_t = Y - APPROACH_OFFSET_M * uy
                    Z_t = max(MIN_Z_CLAMP, Z - APPROACH_OFFSET_M * uz)  # 음수/0 방지

                # TF → map + theta 계산 (오프셋 적용한 목표점 기준)
                if self.tf_ready and X is not None:
                    try:
                        # 원본 사람 위치 (참고 로그)
                        pt_cam_person = PointStamped()
                        pt_cam_person.header.stamp = depth_msg.header.stamp
                        pt_cam_person.header.frame_id = depth_frame_id
                        pt_cam_person.point.x = float(X)
                        pt_cam_person.point.y = float(Y)
                        pt_cam_person.point.z = float(Z)

                        # 오프셋 목표
                        pt_cam_goal = PointStamped()
                        pt_cam_goal.header.stamp = depth_msg.header.stamp
                        pt_cam_goal.header.frame_id = depth_frame_id
                        pt_cam_goal.point.x = float(X_t)
                        pt_cam_goal.point.y = float(Y_t)
                        pt_cam_goal.point.z = float(Z_t)

                        tr = self._tf_lookup_with_fallback(self.target_frame, depth_frame_id, depth_msg.header.stamp)

                        pt_map_person = do_transform_point(pt_cam_person, tr)
                        pt_map_goal = do_transform_point(pt_cam_goal, tr)

                        # 카메라(map) 위치(대략 센서 위치) → tr.translation
                        tx = float(tr.transform.translation.x)
                        ty = float(tr.transform.translation.y)

                        dx = pt_map_goal.point.x - tx
                        dy = pt_map_goal.point.y - ty
                        theta = math.atan2(dy, dx)

                        # 로그: 원본 사람 좌표와 목표 좌표 모두 출력
                        self.get_logger().info(
                            f"[map] {label} person=({pt_map_person.point.x:.2f}, {pt_map_person.point.y:.2f})  "
                            f"goal(+{APPROACH_OFFSET_M:.2f}m toward camera)=({pt_map_goal.point.x:.2f}, {pt_map_goal.point.y:.2f}, θ={theta:.3f}rad)"
                        )

                        # Pose2D 퍼블리시: 오프셋 목표점
                        pose = Pose2D()
                        pose.x = pt_map_goal.point.x
                        pose.y = pt_map_goal.point.y
                        pose.theta = theta
                        self.pose_pub.publish(pose)

                        # ---- 반드시 저장 필요 ----
                        self.last_person_pose_map = pose

                        # === RViz 마커 발행 (참고 코드 스타일) ===
                        # 사람 원래 위치 (붉은 톤)
                        self.publish_marker(
                            pt_map_person.point.x, pt_map_person.point.y, pt_map_person.point.z,
                            label=f"{label}_person", r=1.0, g=0.3, b=0.3, scale=0.22, lifetime_sec=1.0
                        )
                        # 오프셋 목표점 (녹색 톤), 지면 z=0.0
                        self.publish_marker(
                            pt_map_goal.point.x, pt_map_goal.point.y, 0.0,
                            label=f"{label}_goal", r=0.2, g=0.9, b=0.2, scale=0.22, lifetime_sec=1.0
                        )

                    except Exception as e:
                        self.get_logger().warn(f"TF transform failed: {e}")

                # 시각화(깊이 원본 Z 라벨)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(out, (cx_rgb, cy_rgb), 3, (0, 0, 255), -1)
                cv2.putText(out, f"{label} {conf:.2f}  Depth={Z:.2f}m",
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(out, f"infer: {infer_ms:.1f} ms", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if self.annotated_pub is not None:
            self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(out, encoding='bgr8'))
        if self.enable_viewer:
            try:
                cv2.imshow(WINDOW_NAME_RGB, out)
                cv2.waitKey(1)
            except Exception:
                pass

    # lookup with robust time fallbacks
    def _tf_lookup_with_fallback(self, target_frame, source_frame, msg_stamp):
        """
        1) 정확히 msg_stamp 시도
        2) 안되면 0.2s 과거로 백오프
        3) 그래도 안되면 latest (Time())
        """
        # 1) exact stamp
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, msg_stamp,
                timeout=Duration(seconds=self.tf_timeout_sec)
            )
        except Exception:
           pass

        # 2) backoff 0.2s
        try:
            backoff_stamp = rclpy.time.Time(
                seconds=msg_stamp.sec,
                nanoseconds=msg_stamp.nanosec
            ) - Duration(seconds=0.2)
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, backoff_stamp,
                timeout=Duration(seconds=self.tf_timeout_sec)
            )
        except Exception as e2:
            pass

        # 3) latest
        return self.tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time(),
            timeout=Duration(seconds=self.tf_timeout_sec)
        )

    # ---------- utils ----------
    def _depth_to_vis(self, depth_np: np.ndarray, normalize_range_m: float) -> np.ndarray:
        if depth_np is None:
            return np.zeros((480, 640), dtype=np.uint8)
        depth_m = self._raw_depth_to_meters(depth_np)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, normalize_range_m)
        return (depth_m / normalize_range_m * 255.0).astype(np.uint8)

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

    
