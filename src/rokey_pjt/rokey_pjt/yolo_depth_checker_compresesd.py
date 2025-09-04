#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 + Depth 융합 노드
- RGB 프레임에서 YOLOv8n.pt로 객체 탐지
- 각 바운딩 박스 중심(cx, cy) 주변의 Depth를 5x5 윈도우 중앙값으로 측정
- 화면에 박스/라벨/거리(m) 오버레이

설정/운영 팁
- 파라미터로 RGB/Depth/CameraInfo 토픽과 모델 경로, 임계치(conf/iou) 조절 가능
- Depth 인코딩이 uint16(밀리미터) 또는 float32(미터)인 경우를 모두 처리
- 카메라 내부행렬 K(CameraInfo)가 들어오면 로그로 확인(필수는 아님)
"""

import os
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

# YOLOv8 (ultralytics)        pip install ultralytics
from ultralytics import YOLO

# ================================
# 기본 상수
# ================================
DEPTH_TOPIC_DEFAULT = '/robot3/oakd/stereo/image_raw'     # Depth 이미지 토픽
CAMERA_INFO_TOPIC_DEFAULT = '/robot3/oakd/stereo/camera_info'
RGB_TOPIC_DEFAULT = '/robot3/oakd/rgb/image_raw'          # RGB 이미지 토픽(환경에 맞게 변경)
MAX_DEPTH_METERS = 5.0
NORMALIZE_DEPTH_RANGE = 3.0
WINDOW_NAME_RGB = 'YOLO + Depth (bbox center distance)'
WINDOW_NAME_DEPTH = 'Depth Visualization'
# ================================


class YoloDepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth')

        # ----- 공용 유틸 -----
        self.bridge = CvBridge()
        self.K = None                     # CameraInfo 내 내부행렬
        self.depth_cv = None              # 가장 최근 Depth (numpy)
        self.depth_colored = None         # 시각화용 컬러맵 이미지
        self.last_depth_dtype = None      # depth dtype 추적(uint16/float32 등)

        # ----- 파라미터 선언 -----
        # 패키지 내부 models/yolov8n.pt가 기본 경로가 되도록 구성
        pkg_dir = os.path.dirname(os.path.abspath(__file__))            # .../rokey_pjt
        default_model_path = os.path.join(os.path.dirname(pkg_dir), 'models', 'yolov8n.pt')

        self.declare_parameter('rgb_topic', RGB_TOPIC_DEFAULT)
        self.declare_parameter('depth_topic', DEPTH_TOPIC_DEFAULT)
        self.declare_parameter('camera_info_topic', CAMERA_INFO_TOPIC_DEFAULT)
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('conf_thres', 0.5)     # YOLO confidence
        self.declare_parameter('iou_thres', 0.5)      # YOLO IoU
        self.declare_parameter('depth_window', 5)     # 깊이 중앙값 윈도우 크기(홀수 권장)

        # 파라미터 로드
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.iou_thres = float(self.get_parameter('iou_thres').value)
        self.depth_window = int(self.get_parameter('depth_window').value)
        if self.depth_window % 2 == 0:
            self.depth_window += 1  # 짝수면 1 증가시켜 홀수로 맞춤

        # ----- YOLO 로드 -----
        if not os.path.exists(self.model_path):
            self.get_logger().warn(f"YOLO 가중치 파일이 보이지 않습니다: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"YOLO model loaded: {self.model_path}")

        # ----- 구독자 생성 -----
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.caminfo_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        # OpenCV 윈도우
        cv2.namedWindow(WINDOW_NAME_RGB, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME_RGB, 640, 480)  # 창 크기 조절

        cv2.namedWindow(WINDOW_NAME_DEPTH, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME_DEPTH, 640, 480)  # Depth 창도 보기 쉽게

        self.get_logger().info(
            f"Subscribed topics:\n  RGB: {self.rgb_topic}\n  DEPTH: {self.depth_topic}\n  CameraInfo: {self.camera_info_topic}"
        )

    # ---------------------------
    # CameraInfo 콜백
    # ---------------------------
    def camera_info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            self.get_logger().info(f"CameraInfo: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # ---------------------------
    # Depth 콜백: 최신 depth 보관 & 시각화 생성
    # ---------------------------
    def depth_callback(self, msg: Image):
        # depth 이미지를 numpy로 변환
        depth_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_cv = np.array(depth_np, copy=False)
        self.last_depth_dtype = self.depth_cv.dtype

        # 시각화용 컬러맵 구성(보기 편하게 0~NORMALIZE_DEPTH_RANGE m 스케일)
        depth_vis = self._depth_to_vis(self.depth_cv, NORMALIZE_DEPTH_RANGE)
        self.depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow(WINDOW_NAME_DEPTH, self.depth_colored)
        cv2.waitKey(1)

    # ---------------------------
    # RGB 콜백: YOLO 추론 + 중심점 거리 계산/표시
    # ---------------------------
    def rgb_callback(self, msg: Image):
        # YOLO 입력용 BGR 프레임
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame_bgr.shape[:2]

        # YOLO 추론
        results = self.model(
            frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False
        )[0]

        # 탐지 결과 순회
        if results.boxes is not None:
            for box in results.boxes:
                # 박스 좌표/클래스/신뢰도
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, xyxy)
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                label = self.model.names.get(cls_id, str(cls_id))

                # 박스 중심 좌표 (화면 밖 방지용 클램프)
                cx = max(0, min((x1 + x2) // 2, w - 1))
                cy = max(0, min((y1 + y2) // 2, h - 1))

                # 중심점 주변 윈도우에서 Depth(m) 중앙값 추정
                dist_m = self._get_depth_median_m(cx, cy)

                # 시각화(박스/중심/텍스트)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame_bgr, (cx, cy), 3, (0, 0, 255), -1)

                dist_text = "d=? m" if (dist_m is None or math.isnan(dist_m)) else f"d={dist_m:.2f} m"
                text = f"{label} {conf:.2f} | {dist_text}"
                self._draw_label(frame_bgr, text, x1, y1 - 5)

        # 보여주기
        cv2.imshow(WINDOW_NAME_RGB, frame_bgr)
        cv2.waitKey(1)

    # ===========================
    # 유틸 함수들
    # ===========================
    def _depth_to_vis(self, depth_np: np.ndarray, normalize_range_m: float) -> np.ndarray:
        """
        Depth numpy를 0~255 범위의 8bit 이미지로 변환(컬러맵 전에 사용)
        - normalize_range_m: 시각화 상한(m)
        """
        if depth_np is None:
            return np.zeros((480, 640), dtype=np.uint8)

        # 원시 depth → 미터 단위로 변환
        depth_m = self._raw_depth_to_meters(depth_np)

        # NaN/Inf/음수 처리 후 클립
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, normalize_range_m)

        # 0~255 스케일
        vis = (depth_m / normalize_range_m * 255.0).astype(np.uint8)
        return vis

    def _raw_depth_to_meters(self, depth_np: np.ndarray) -> np.ndarray:
        """
        센서에서 들어온 depth 배열을 '미터(m)' 스케일로 변환
        - uint16/uint32: 밀리미터 단위로 가정 → /1000
        - float32/float64: 이미 미터로 가정 → 그대로 사용
        """
        if depth_np is None:
            return None

        if np.issubdtype(depth_np.dtype, np.floating):
            return depth_np.astype(np.float32)
        else:
            # 보편적 깊이 토픽(16UC1) 가정: mm→m
            return depth_np.astype(np.float32) / 1000.0

    def _get_depth_median_m(self, x: int, y: int) -> float:
        """
        (x, y) 주변 depth_window×depth_window 패치에서
        0/NaN을 제외한 값의 중앙값을 미터 단위로 반환.
        유효 샘플이 없으면 None.
        """
        if self.depth_cv is None:
            return None

        # 미터 변환
        depth_m = self._raw_depth_to_meters(self.depth_cv)
        if depth_m is None:
            return None

        h, w = depth_m.shape[:2]
        r = self.depth_window // 2
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w, x + r + 1), min(h, y + r + 1)

        patch = depth_m[y0:y1, x0:x1].reshape(-1)
        # 유효 값만 선택(>0)
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _draw_label(self, img: np.ndarray, text: str, x: int, y: int):
        """
        가독성 있는 바운딩 박스 라벨 렌더링(배경 박스 + 텍스트)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        # 배경 박스
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), (0, 0, 0), -1)
        # 텍스트
        cv2.putText(img, text, (x + 2, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    rclpy.init()
    node = YoloDepthNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()