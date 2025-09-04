#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip install ultralytics opencv-python pyyaml

import os
import time
import math
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from collections import deque

# ========= 사용자 설정 =========
MODEL_PATH = "/home/rokey/rokey_ws/src/yolov8/cvs/card/weights/best.pt"    # OBB 모델 경로
CAM_DEVICE = "/dev/video2"                      # USB 웹캠 경로
CAM_W, CAM_H, CAM_FPS = 640, 640, 30
CONF, IOU = 0.25, 0.5
IMG_SIZE = 640
H_YAML = "/home/rokey/rokey_ws/src/yolov8/homography.yaml"                                    # homography.yaml 경로 (없으면 cm 표시는 비활성화)
ROI_POINTS_BASE = np.array([
    [169, 132],  # 좌상
    [571, 142],  # 우상
    [597, 338],  # 좌하
    [141, 327],  # 우하

], dtype=np.float32)
BASE_W, BASE_H = 640, 480
BUFFER_SIZE = 10  # 최근 5프레임 사용
# =================================

def open_camera(dev=CAM_DEVICE, w=CAM_W, h=CAM_H, fps=CAM_FPS):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if isinstance(dev, str) else cv2.VideoCapture(int(dev), cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERR] Camera open failed: {dev}")
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] requested {w}x{h}@{fps}, actual {aw}x{ah}@{af:.0f}")
    return cap

def sort_quad_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]

def xywhr_to_corners(x, y, w, h, r):
    dx, dy = w / 2.0, h / 2.0
    rect = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    c, s = math.cos(r), math.sin(r)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rot = rect @ R.T
    rot[:, 0] += x
    rot[:, 1] += y
    return rot

def load_H(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return np.array(data["H"], dtype=np.float64)

def obb_to_metrics_cm(quad_xy, H):
    pts = sort_quad_clockwise(quad_xy).astype(np.float32)[None, ...]
    bev = cv2.perspectiveTransform(pts, H)[0]
    edges = np.linalg.norm(np.roll(bev, -1, axis=0) - bev, axis=1)
    L, W = float(edges.max()), float(edges.min())
    i = int(np.argmax(edges))
    p0, p1 = bev[i], bev[(i + 1) % 4]
    yaw = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    if yaw > 180: yaw -= 360
    if yaw <= -180: yaw += 360
    return L, W, yaw

def main():
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    H = load_H(H_YAML) if H_YAML else None

    model = YOLO(MODEL_PATH)
    cap = open_camera()
    if cap is None:
        return

    # 첫 프레임을 읽어서 실제 카메라 해상도 확인
    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit("[ERR] 첫 프레임 획득 실패")
    frame_h, frame_w = frame.shape[:2]

    # ROI 좌표를 실제 해상도에 맞게 스케일링
    scale_x = frame_w / float(BASE_W)
    scale_y = frame_h / float(BASE_H)
    ROI_POINTS = np.stack([
        np.array([int(x * scale_x), int(y * scale_y)], dtype=np.int32)
        for (x, y) in ROI_POINTS_BASE
    ])

    # 프레임 크기에 맞춰 마스크 생성
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask, [ROI_POINTS], 255)

    # 최근 5프레임 측정값 버퍼
    len_buf, wid_buf, yaw_buf = deque(maxlen=BUFFER_SIZE), deque(maxlen=BUFFER_SIZE), deque(maxlen=BUFFER_SIZE)

    show_annot = True
    fps = 0.0
    t0 = time.time()

    print("[INFO] q: quit | a: annotated/raw toggle")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # 프레임 크기 변경 시 ROI와 마스크 재계산
        h, w = frame.shape[:2]
        if mask.shape[0] != h or mask.shape[1] != w:
            scale_x = w / float(BASE_W)
            scale_y = h / float(BASE_H)
            ROI_POINTS = np.stack([
                np.array([int(x * scale_x), int(y * scale_y)], dtype=np.int32)
                for (x, y) in ROI_POINTS_BASE
            ])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [ROI_POINTS], 255)

        # ROI 내부만 추론하도록 마스크 적용
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # YOLO OBB 추론
        res = model.predict(source=roi_frame, conf=CONF, iou=IOU, imgsz=IMG_SIZE,
                            device=0, task="obb", verbose=False, stream=False)[0]

        # OBB 좌표 가져오기
        quads = None
        if hasattr(res, "obb") and getattr(res.obb, "xyxyxyxy", None) is not None:
            quads = res.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        elif hasattr(res, "obb") and getattr(res.obb, "xywhr", None) is not None:
            xywhr = res.obb.xywhr.cpu().numpy()
            if len(xywhr):
                quads = np.stack([xywhr_to_corners(*row) for row in xywhr])

        view = frame.copy()
        cv2.polylines(view, [ROI_POINTS], True, (0, 255, 0), 2)

        if quads is not None and len(quads) > 0:
            for q in quads:
                pts = sort_quad_clockwise(q).astype(int)
                cv2.polylines(view, [pts], True, (0, 255, 255), 2)

                if H is not None:
                    # 새 측정값 추가
                    L, W, yaw = obb_to_metrics_cm(q, H)
                    len_buf.append(L)
                    wid_buf.append(W)
                    yaw_buf.append(yaw)

                    # 최근 5프레임의 중앙값 사용
                    Lm = np.median(len_buf)
                    Wm = np.median(wid_buf)
                    Yaw = np.median(yaw_buf)

                    label = f"L:{Lm:.1f}cm W:{Wm:.1f}cm yaw:{Yaw:.1f}°"
                else:
                    label = "OBB detected"

                p = pts[0]
                cv2.putText(view, label, (p[0], max(0, p[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 230, 50), 2)
        else:
            cv2.putText(view, "No OBB in ROI", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 255), 2)

        # FPS 계산
        t1 = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t1 - t0, 1e-6))
        t0 = t1
        cv2.putText(view, f"FPS:{fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("USB OBB ROI", view if show_annot else frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('a'):
            show_annot = not show_annot

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
