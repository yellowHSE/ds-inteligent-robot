#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBB 차량 감지 노드 (Sticky + ClassGate + Slow Fade-out + Stationary Edge TX)
- yaw은 라디안으로 계산/전송 (음수면 π 더해서 (0, π] 범위)
- "정지 상태"로 진입하는 순간에만 /entry & /car_size 1회 전송(원샷)
- 바운딩 박스 라벨 제거, 화면 우측 하단 패널에 state(STOP/MOVING), size(compact/general), yaw(rad) 표시
"""

import os, time, math, cv2, yaml
import numpy as np
from collections import deque, Counter
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from parking_msgs.srv import Entry, CarSize

# ===== 설정 =====
CAM_DEVICE = "/dev/video2"            # "/dev/video2" 또는 정수 인덱스(예: 2)
CAM_W, CAM_H, CAM_FPS = 640, 480, 25  # 안전 해상도
TIMER_HZ = 30.0

CONF, IOU = 0.25, 0.5
IMG_SIZE = 640

ROI_POINTS_BASE = np.array([[136,118],[607,124],[628,395],[111,395]], dtype=np.float32)
BASE_W, BASE_H = 640, 480

# 버퍼/안정화
BUFFER_SIZE = 10
STAB_WINDOW = 10
STAB_ON     = 7
STAB_OFF    = 3

# Sticky(그레이스)
STICKY_LOSS_FRAMES = 10
REACQ_DIST_CM      = 120.0
REACQ_YAW_RAD      = math.radians(25.0)

# 클래스 일관성 게이트
CLASS_GATE_ENABLED   = True
CLASS_GATE_HISTORY   = 10
CLASS_GATE_MIN_COUNT = 3

# 필터/조건
STRICT_METRICS  = False
MIN_CONF        = 0.10
CLASS_WHITELIST = None

MIN_LENGTH_CM       = 60.0
MAX_LENGTH_CM       = 1000.0
MAX_YAW_ABS_DEG     = 120.0  # (표시용 한계)
MIN_BUF_FOR_METRICS = 3

# --- 정지 판별 파라미터 ---
STOP_WINDOW = 10  # 최근 N프레임으로 판정
STOP_EPS_TRANS_CM_STEP = 2.0                 # 연속 프레임 간 이동 허용(cm)
STOP_EPS_TRANS_CM_SUM  = 20.0                # 창 전체 이동 합(cm)
STOP_EPS_YAW_RAD_STEP  = math.radians(1.5)   # 연속 프레임 간 회전 허용(rad)
STOP_EPS_PIX_STEP      = 3.0                 # 이미지 픽셀 기준 보조 임계(센터 이동)
STOP_MIN_IOU_AABB      = 0.90                # 연속 AABB IoU 하한

# 서비스
ENTRY_SERVICE_NAME     = "/entry"
CARSIZE_SERVICE_NAME   = "/car_size"
SERVICE_TIMEOUT_SEC    = 2.0
CARSIZE_RETRY_COOLDOWN = 0.5
# =================

def sort_quad_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    return pts[np.argsort(ang)]

def xywhr_to_corners(x,y,w,h,r):
    dx, dy = w/2.0, h/2.0
    rect = np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]], dtype=np.float32)
    c, s = math.cos(r), math.sin(r)
    R = np.array([[c,-s],[s,c]], dtype=np.float32)
    rot = rect @ R.T
    rot[:,0]+=x; rot[:,1]+=y
    return rot

def load_H(yaml_path):
    with open(yaml_path,"r") as f:
        data = yaml.safe_load(f)
    return np.array(data["H"], dtype=np.float64)

def angle_diff_rad(a, b):
    """라디안 각 차이 (0..π)"""
    d = abs(a - b)
    return min(d, 2*math.pi - d)

def obb_to_metrics_cm(quad_xy, H):
    """
    입력: 이미지좌표계 사각형 4점
    처리: BEV로 투영 후, 가장 긴 변을 차량 길이로 보고 그 방향을 yaw로 정의
    yaw 계산/정규화: atan2→[-π,π], 음수면 π 더해 (0, π] 범위로 보정
    """
    pts = sort_quad_clockwise(quad_xy).astype(np.float32)[None,...]
    bev = cv2.perspectiveTransform(pts, H)[0]
    edges = np.linalg.norm(np.roll(bev,-1,axis=0)-bev, axis=1)
    L, W = float(edges.max()), float(edges.min())
    i = int(np.argmax(edges)); p0, p1 = bev[i], bev[(i+1)%4]

    yaw = math.atan2(p1[1]-p0[1], p1[0]-p0[0])  # [-π, π]
    if yaw <= -math.pi:
        yaw += 2*math.pi
    if yaw < 0.0:
        yaw += math.pi

    return L, W, yaw, bev  # yaw in radians, range (0, π]

def classify_length_cm(L):
    return "compact" if L < 50.0 else "general"

def majority_id(seq):
    if not seq:
        return None, 0
    c = Counter(seq)
    (val, cnt) = max(c.items(), key=lambda kv: (kv[1], kv[0]))
    return val, cnt

def aabb_from_pts(pts_int):  # pts_int: (4,2) int
    xs = pts_int[:,0]; ys = pts_int[:,1]
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def iou_aabb(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def draw_yaw_compass(img, yaw_rad, yaw0_rad=None, center=None, radius=56):
    """
    우측 상단에 0~180° 반원 게이지 + 현재 yaw 포인터, 기준 yaw0 대비 Δ표시(있으면).
    yaw_rad / yaw0_rad 는 라디안.
    """
    h, w = img.shape[:2]
    cx, cy = center if center is not None else (w - 90, 100)
    r = int(radius)

    # 반투명 배경
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r+10, (0,0,0), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # 반원 틱(0,45,90,135,180)
    for deg in (0,45,90,135,180):
        a = math.radians(deg)
        x1 = int(cx + r*math.cos(a)); y1 = int(cy - r*math.sin(a))
        x2 = int(cx + (r-8)*math.cos(a)); y2 = int(cy - (r-8)*math.sin(a))
        cv2.line(img, (x1,y1), (x2,y2), (160,160,160), 2)

    # 테두리(반원)
    cv2.ellipse(img, (cx,cy), (r,r), 0, 0, 180, (200,200,200), 2)

    # 현재 yaw 포인터
    a = math.pi - yaw_rad   # ← 방향 반전(0~π 반원에서 좌우 반전)
    x = int(cx + (r-12)*math.cos(a)); y = int(cy - (r-12)*math.sin(a))
    cv2.circle(img, (x,y), 5, (0,255,255), -1)
    cv2.line(img, (cx,cy), (x,y), (0,255,255), 2)

    # 텍스트: yaw 및 Δyaw(있으면)
    yaw_deg = math.degrees(yaw_rad)
    label = f"yaw {yaw_deg:.1f}°"
    # cv2.putText(img, label, (cx - r, cy + r + 18),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,230,50), 2, cv2.LINE_AA)

    if yaw0_rad is not None:
        d = math.degrees(min(abs(yaw_rad - yaw0_rad), 2*math.pi - abs(yaw_rad - yaw0_rad)))
        # cv2.putText(img, f"Δ{d:.1f}°", (cx - r, cy + r + 38),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,230,50), 2, cv2.LINE_AA)

def draw_bottom_right_panel(img, lines, margin=20, pad=10, alpha=0.35,
                            font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, thickness=2):
    """오른쪽 아래에 반투명 패널로 여러 줄 텍스트를 그린다."""
    h, w = img.shape[:2]
    if not lines:
        return
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    box_w = max(s[0] for s in sizes) + pad*2
    line_h = max(s[1] for s in sizes)
    gap = 6
    box_h = line_h*len(lines) + gap*(len(lines)-1) + pad*2

    x1 = w - margin - box_w
    y1 = h - margin - box_h
    x2 = w - margin
    y2 = h - margin

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    y = y1 + pad + line_h
    for t in lines:
        cv2.putText(img, t, (x1 + pad, y), font, scale, (50, 230, 50), thickness, cv2.LINE_AA)
        y += line_h + gap

class ObbEntryCarSizeNode(Node):
    def __init__(self):
        super().__init__("obb_entry_carsize_node")

        # --- Services ---
        self.entry_cli   = self.create_client(Entry, ENTRY_SERVICE_NAME)
        self.carsize_cli = self.create_client(CarSize, CARSIZE_SERVICE_NAME)
        self.get_logger().info(f"Waiting for service: {ENTRY_SERVICE_NAME}")
        while not self.entry_cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("Entry service not available yet...")
        self.get_logger().info(f"Waiting for service: {CARSIZE_SERVICE_NAME}")
        while not self.carsize_cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("CarSize service not available yet...")

        # --- PATH ----
        pkg_root = Path(__file__).resolve().parent
        MODEL_PATH = str(pkg_root / "models" / "obb_car_size.pt")
        H_YAML = str(pkg_root / "models" / "homography.yaml")

        # --- Model & names ---
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        raw_names = getattr(self.model, "names", {})
        if isinstance(raw_names, dict):
            self.names = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, (list, tuple)):
            self.names = {i: str(n) for i, n in enumerate(raw_names)}
        else:
            self.names = {}

        self.H = load_H(H_YAML) if H_YAML else None

        # --- Camera ---
        self.cap = self._open_camera(CAM_DEVICE, CAM_W, CAM_H, CAM_FPS)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("첫 프레임 획득 실패")
        self.frame_h, self.frame_w = frame.shape[:2]
        self._update_roi_mask(self.frame_w, self.frame_h)

        # --- Buffers & states ---
        self.len_buf = deque(maxlen=BUFFER_SIZE)
        self.wid_buf = deque(maxlen=BUFFER_SIZE)
        self.yaw_buf = deque(maxlen=BUFFER_SIZE)   # rad

        self.stab_score = 0
        self.det_state  = False

        self.yaw_for_vis = None        # 컴퍼스에 표시할 최신 yaw(rad)
        self.yaw_baseline = None       # STOP 순간 기준 yaw(rad)  

        # 서비스/전송 래치
        self.entry_sent_latch = False
        self.carsize_sent = False
        self.carsize_last_try = 0.0
        self.stop_sent_latch = True

        # Sticky
        self.track_active = False
        self.miss_count = 0
        self.last_center = None
        self.last_yaw = None          # rad

        # Class gate
        self.class_hist = deque(maxlen=CLASS_GATE_HISTORY)
        self.last_major_cls = None

        # 페이드/표시
        self.last_main_pts = None
        self.last_size_str = None

        # 정지 판별 버퍼/상태
        self.center_buf = deque(maxlen=STOP_WINDOW)   # (x_cm, y_cm)
        self.yaw_hist   = deque(maxlen=STOP_WINDOW)   # rad
        self.img_center_buf = deque(maxlen=STOP_WINDOW)  # 이미지 px
        self.aabb_buf       = deque(maxlen=STOP_WINDOW)  # (xmin,ymin,xmax,ymax)
        self.stationary_state = False

        self.show_annot = True
        self.fps = 0.0
        self._t_prev = time.time()

        self.timer = self.create_timer(1.0/TIMER_HZ, self._loop)
        self.get_logger().info("✅ node started (q: quit / a: overlay toggle)")

    # ---------- Camera ----------
    def _open_camera(self, dev, w, h, fps):
        # 여러 백엔드/포맷 폴백으로 카메라 오픈 안정화
        candidates = []
        if isinstance(dev, str) and dev.startswith("/dev/video"):
            candidates.append( (dev, cv2.CAP_V4L2) )
            candidates.append( (dev, cv2.CAP_ANY) )
            candidates.append( (dev, cv2.CAP_GSTREAMER) )
            try:
                idx = int(dev.replace("/dev/video", ""))
                candidates.append( (idx, cv2.CAP_V4L2) )
                candidates.append( (idx, cv2.CAP_ANY) )
            except Exception:
                pass
        else:
            candidates.append( (int(dev), cv2.CAP_V4L2) )
            candidates.append( (int(dev), cv2.CAP_ANY) )
            candidates.append( (int(dev), cv2.CAP_GSTREAMER) )

        last_err = "unknown"
        for target, backend in candidates:
            cap = cv2.VideoCapture(target, backend)
            if not cap.isOpened():
                last_err = f"backend={backend} open failed"
                continue

            fourccs = ["MJPG", "YUYV", None]
            for fcc in fourccs:
                if fcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
                cap.set(cv2.CAP_PROP_FPS, int(fps))
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                af = cap.get(cv2.CAP_PROP_FPS)
                self.get_logger().info(f"[CAM try] {target} backend={backend}, fourcc={fcc}, "
                                       f"requested {w}x{h}@{fps}, actual {aw}x{ah}@{af:.0f}")

                # 워밍업
                warm_ok = False
                for _ in range(10):
                    ok, _frm = cap.read()
                    if ok and _frm is not None and _frm.size > 0:
                        warm_ok = True
                        break
                    time.sleep(0.03)

                if not warm_ok:
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    for _ in range(10):
                        ok, _frm = cap.read()
                        if ok and _frm is not None and _frm.size > 0:
                            warm_ok = True
                            break
                        time.sleep(0.03)

                if warm_ok:
                    self.get_logger().info(f"[CAM OK] {target} backend={backend}, fourcc={fcc}")
                    return cap

            cap.release()

        raise RuntimeError(f"Camera open failed: {dev} ({last_err})")

    def _update_roi_mask(self, w, h):
        scale_x = w/float(BASE_W); scale_y = h/float(BASE_H)
        self.ROI_POINTS = np.stack([np.array([int(x*scale_x), int(y*scale_y)], dtype=np.int32)
                                    for (x,y) in ROI_POINTS_BASE])
        self.mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self.mask, [self.ROI_POINTS], 255)

    # ---------- Service senders ----------
    def _send_entry_angle(self, angle_rad: float):
        """Entry.srv(angle: float32, radians) 전송"""
        if not self.entry_cli.service_is_ready():
            return
        req = Entry.Request()
        if hasattr(req, "angle"):
            req.angle = float(angle_rad)
        else:
            try:
                fields = req.get_fields_and_field_types()
                for name, typ in fields.items():
                    if "float" in typ:
                        setattr(req, name, float(angle_rad))
                        break
            except Exception:
                pass

        fut = self.entry_cli.call_async(req)
        def _done_cb(f):
            try:
                resp = f.result()
                if getattr(resp, "success", True):
                    self.entry_sent_latch = True
                    self.get_logger().info(f"Entry.srv sent (angle={angle_rad:.3f} rad)")
                else:
                    self.get_logger().warn("Entry.srv responded success=False")
            except Exception as e:
                self.get_logger().warn(f"Entry.srv failed: {e!r}")
        fut.add_done_callback(_done_cb)

    def _try_send_carsize(self, size_str: str):
        """CarSize.srv(size: string) 전송(성공 시까지 쿨다운 재시도)"""
        now = time.time()
        if now - self.carsize_last_try < CARSIZE_RETRY_COOLDOWN:
            return
        self.carsize_last_try = now
        if not self.carsize_cli.service_is_ready():
            return

        req = CarSize.Request()
        if hasattr(req, "size"):
            req.size = str(size_str)

        fut = self.carsize_cli.call_async(req)
        def _done_cb(f):
            try:
                resp = f.result()
                if getattr(resp, "success", True):
                    self.carsize_sent = True
                    self.get_logger().info(f"CarSize.srv sent (size='{size_str}')")
                else:
                    self.get_logger().warn("CarSize.srv responded success=False")
            except Exception as e:
                self.get_logger().warn(f"CarSize.srv failed: {e!r}")
        fut.add_done_callback(_done_cb)

    # ---------- Helpers ----------
    def _reset_track(self):
        self.track_active = False
        self.miss_count = 0
        self.last_center = None
        self.last_yaw = None

        self.len_buf.clear(); self.wid_buf.clear(); self.yaw_buf.clear()
        self.class_hist.clear()
        self.last_major_cls = None

        # 정지 관련 버퍼도 리셋
        self.center_buf.clear()
        self.yaw_hist.clear()
        self.img_center_buf.clear()
        self.aabb_buf.clear()
        self.stationary_state = False
        self.stop_sent_latch = False

    def _is_stationary(self) -> bool:
        """
        최근 STOP_WINDOW 프레임이 모두 '매우 미세한 변화'였는지 검사.
        - 연속 프레임 간 이동/회전이 모두 step 임계 이하
        - 창 전체 이동 합이 sum 임계 이하
        - 연속 AABB IoU가 모두 HIGH (형상/크기 안정)
        - 이미지 픽셀 기준 보조 임계도 만족
        """
        N = min(len(self.center_buf), len(self.yaw_hist),
                len(self.img_center_buf), len(self.aabb_buf))
        if N < STOP_WINDOW:
            return False

        step_d_cm = []
        step_dyaw = []
        step_d_px = []
        step_iou  = []

        for i in range(1, N):
            # BEV(cm)
            x0, y0 = self.center_buf[i-1]; x1, y1 = self.center_buf[i]
            d_cm = math.hypot(x1 - x0, y1 - y0)
            step_d_cm.append(d_cm)

            dy = angle_diff_rad(self.yaw_hist[i], self.yaw_hist[i-1])
            step_dyaw.append(dy)

            # 이미지(px)
            ix0, iy0 = self.img_center_buf[i-1]; ix1, iy1 = self.img_center_buf[i]
            d_px = math.hypot(ix1 - ix0, iy1 - iy0)
            step_d_px.append(d_px)

            # 형상 안정성(AABB IoU)
            iou = iou_aabb(self.aabb_buf[i], self.aabb_buf[i-1])
            step_iou.append(iou)

        cond_step = all( (d <= STOP_EPS_TRANS_CM_STEP) and
                         (dy <= STOP_EPS_YAW_RAD_STEP) and
                         (dp <= STOP_EPS_PIX_STEP)
                         for d, dy, dp in zip(step_d_cm, step_dyaw, step_d_px) )

        cond_sum = (sum(step_d_cm) <= STOP_EPS_TRANS_CM_SUM)
        cond_iou = (min(step_iou) >= STOP_MIN_IOU_AABB)
        return cond_step and cond_sum and cond_iou

    # ---------- Main loop ----------
    def _loop(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn("Camera read failed")
            return

        h, w = frame.shape[:2]
        if self.mask.shape[:2] != (h, w):
            self._update_roi_mask(w, h)

        roi_frame = cv2.bitwise_and(frame, frame, mask=self.mask)

        # YOLO OBB
        res0 = self.model.predict(
            source=roi_frame, conf=CONF, iou=IOU, imgsz=IMG_SIZE,
            device=0, task="obb", verbose=False, stream=False
        )[0]

        quads = None; confs = None; clss = None
        if hasattr(res0,"obb") and getattr(res0.obb,"xyxyxyxy",None) is not None:
            quads = res0.obb.xyxyxyxy.cpu().numpy().reshape(-1,4,2)
            confs = res0.obb.conf.cpu().numpy() if getattr(res0.obb,"conf",None) is not None else None
            clss  = res0.obb.cls.cpu().numpy().astype(int) if getattr(res0.obb,"cls",None) is not None else None
        elif hasattr(res0,"obb") and getattr(res0.obb,"xywhr",None) is not None:
            xywhr = res0.obb.xywhr.cpu().numpy()
            if len(xywhr):
                quads = np.stack([xywhr_to_corners(*row) for row in xywhr])
                confs = res0.obb.conf.cpu().numpy() if getattr(res0.obb,"conf",None) is not None else None
                clss  = res0.obb.cls.cpu().numpy().astype(int) if getattr(res0.obb,"cls",None) is not None else None

        # YOLO 시각화(ROI)
        try:
            yolo_vis = res0.plot()
            cv2.imshow("YOLO OBB (ROI)", yolo_vis)
        except Exception:
            pass

        view = frame.copy()
        cv2.polylines(view, [self.ROI_POINTS], True, (0,255,0), 2)

        # ---- 검출/계산 ----
        panel_values = None  # ("size_str", yaw_rad)
        detected_this_frame = False
        size_str = None
        angle_val = None
        cur_center = None; cur_yaw = None
        cur_cls_id = None
        main_pts = None

        indices = list(range(len(quads))) if (quads is not None and len(quads) > 0) else []
        if indices and confs is not None:
            indices = [i for i in indices if confs[i] >= MIN_CONF]
        if indices and CLASS_WHITELIST and clss is not None:
            indices = [i for i in indices if int(clss[i]) in CLASS_WHITELIST]

        # 얇은 라벨(모든 박스: 클래스+conf)
        # for i in indices:
        #     pts_all = sort_quad_clockwise(quads[i]).astype(int)
        #     cv2.polylines(view, [pts_all], True, (60,180,255), 1)
        #     cls_id = int(clss[i]) if (clss is not None and i < len(clss)) else -1
        #     cls_name = self.names.get(cls_id, f"id{cls_id}" if cls_id >= 0 else "obj")
        #     conf_txt = f"{confs[i]:.2f}" if (confs is not None and i < len(confs)) else ""
        #     label_txt = f"{cls_name} {conf_txt}".strip()
        #     org = (pts_all[0][0], max(0, pts_all[0][1] - 5))
        #     cv2.putText(view, label_txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,200,255), 1)

        if len(indices) > 0 and self.H is not None:
            # 가장 큰 박스
            pick = max(indices, key=lambda i: cv2.contourArea(sort_quad_clockwise(quads[i]).astype(np.float32)))
            q = quads[pick]
            pts = sort_quad_clockwise(q).astype(int)
            main_pts = pts.copy()
            # 기존
            cv2.polylines(view, [pts], True, (0,255,255), 2)

            # === [추가] 짧은 변(=width) 두 중점을 지나는 '직선'을 프레임 끝까지 연장해서 그리기 ===
            # pts: 시계방향 정렬된 4점 (int)
            vecs = np.roll(pts, -1, axis=0) - pts                     # 각 변 벡터
            lens = np.linalg.norm(vecs.astype(np.float32), axis=1)    # 각 변 길이
            short_idx = np.argsort(lens)[:2]                          # 가장 짧은 변 2개 인덱스

            # 각 변의 중점 계산 (float로 계산 후 사용)
            m1 = (pts[short_idx[0]] + pts[(short_idx[0] + 1) % 4]) * 0.5
            m2 = (pts[short_idx[1]] + pts[(short_idx[1] + 1) % 4]) * 0.5
            m1 = m1.astype(np.float32)
            m2 = m2.astype(np.float32)

            # 프레임 경계와의 교점을 구해 직선을 확장 (이미지 좌표계)
            h, w = view.shape[:2]
            x1, y1 = float(m1[0]), float(m1[1])
            x2, y2 = float(m2[0]), float(m2[1])
            dx, dy = (x2 - x1), (y2 - y1)

            # 교점 후보 계산 함수들
            cand = []

            eps = 1e-9
            # x = 0
            if abs(dx) > eps:
                y = y1 + (0.0 - x1) * (dy / dx)
                if 0.0 <= y <= (h - 1):
                    cand.append((0, int(round(y))))
            # x = w-1
            if abs(dx) > eps:
                y = y1 + ((w - 1.0) - x1) * (dy / dx)
                if 0.0 <= y <= (h - 1):
                    cand.append((w - 1, int(round(y))))
            # y = 0
            if abs(dy) > eps:
                x = x1 + (0.0 - y1) * (dx / dy)
                if 0.0 <= x <= (w - 1):
                    cand.append((int(round(x)), 0))
            # y = h-1
            if abs(dy) > eps:
                x = x1 + ((h - 1.0) - y1) * (dx / dy)
                if 0.0 <= x <= (w - 1):
                    cand.append((int(round(x)), h - 1))

            # 중복 제거하고 두 점만 선택
            # (수직/수평에 가까운 경우에도 안정적으로 두 점을 얻기 위함)
            uniq = []
            for p in cand:
                if not uniq or (p[0] != uniq[-1][0] or p[1] != uniq[-1][1]):
                    uniq.append(p)
                if len(uniq) == 2:
                    break

            # 교점이 2개면 직선 그리기, 부족하면 중점간 선분으로 폴백
            if len(uniq) == 2:
                cv2.line(view, uniq[0], uniq[1], (255, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.line(view, (int(round(m1[0])), int(round(m1[1]))),
                            (int(round(m2[0])), int(round(m2[1]))),
                            (255, 0, 255), 2, cv2.LINE_AA)

            # (선택) 중점 표시
            cv2.circle(view, (int(round(m1[0])), int(round(m1[1]))), 4, (255, 0, 255), -1)
            cv2.circle(view, (int(round(m2[0])), int(round(m2[1]))), 4, (255, 0, 255), -1)
            # =============================================================================


            L, W, yaw_rad, bev = obb_to_metrics_cm(q, self.H)
            self.len_buf.append(L); self.wid_buf.append(W); self.yaw_buf.append(yaw_rad)

            Lm = float(np.median(self.len_buf)) if len(self.len_buf) else 0.0
            Wm = float(np.median(self.wid_buf)) if len(self.wid_buf) else 0.0
            Yaw = float(np.median(self.yaw_buf)) if len(self.yaw_buf) else 0.0  # rad

            self.yaw_for_vis = Yaw  # 컴퍼스에 쓸 값 갱신

            # 클래스
            pick_cls_id = int(clss[pick]) if (clss is not None and pick < len(clss)) else -1
            pick_cls_name = self.names.get(pick_cls_id, f"id{pick_cls_id}" if pick_cls_id >= 0 else "obj")
            cur_cls_id = pick_cls_id

            # 현재 BEV 중심/각
            cur_center = (float(bev[:,0].mean()), float(bev[:,1].mean()))
            cur_yaw = Yaw

            # 이미지 좌표 중심/경계박스 기록(보조 판정용)
            center_img = (float(pts[:,0].mean()), float(pts[:,1].mean()))
            self.img_center_buf.append(center_img)
            self.aabb_buf.append(aabb_from_pts(pts))

            detected_this_frame = True

            # 전송/표시 페이로드
            size_str  = classify_length_cm(Lm)
            angle_val = float(Yaw)
            self.last_size_str = size_str
            panel_values = (size_str, Yaw)

            # 클래스 히스토리 갱신
            self.class_hist.append(cur_cls_id)
            major_cls, cnt = majority_id(self.class_hist)
            self.last_major_cls = major_cls
            maj_name = self.names.get(major_cls, str(major_cls)) if major_cls is not None else "N/A"
            # cv2.putText(view, f"CLSmaj:{maj_name}({cnt})", (10, 120),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,230,80), 2)

            # Sticky 재획득 판정
            if not self.track_active:
                self.track_active = True
                self.miss_count = 0
                self.last_center = cur_center
                self.last_yaw = cur_yaw
            else:
                if self.miss_count > 0 and self.last_center is not None and self.last_yaw is not None:
                    dist = ((cur_center[0]-self.last_center[0])**2 + (cur_center[1]-self.last_center[1])**2) ** 0.5
                    dyaw = angle_diff_rad(cur_yaw, self.last_yaw)
                    same_geo = (dist <= REACQ_DIST_CM) and (dyaw <= REACQ_YAW_RAD)

                    same_cls = True
                    if CLASS_GATE_ENABLED:
                        major_cls, cnt = majority_id(self.class_hist)
                        if (major_cls is not None) and (cnt >= CLASS_GATE_MIN_COUNT) and (cur_cls_id is not None):
                            same_cls = (cur_cls_id == major_cls)

                    if not (same_geo and same_cls):
                        self.get_logger().info(
                            f"[reacq] NEW vehicle: d={dist:.1f}cm, dYaw={dyaw:.3f}rad, cls_ok={same_cls}"
                        )
                        self._reset_track()
                        self.track_active = True

                self.miss_count = 0
                self.last_center = cur_center
                self.last_yaw = cur_yaw

            # ---- 정지 판별용 버퍼 갱신 (실제 검출된 프레임만) ----
            self.center_buf.append(cur_center)
            self.yaw_hist.append(cur_yaw)

        else:
            cv2.putText(view, "No OBB in ROI", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20,20,255), 2)

        # ---- Sticky 그레이스 처리 ----
        effective_detected = detected_this_frame
        if not detected_this_frame:
            if self.track_active:
                self.miss_count += 1
                if self.miss_count <= STICKY_LOSS_FRAMES:
                    effective_detected = True
                    if self.last_main_pts is not None:
                        main_pts = self.last_main_pts.copy()
                else:
                    self._reset_track()
            else:
                effective_detected = False

        # 안정 점수 업데이트(히스테리시스)
        prev_state = self.det_state
        if effective_detected:
            self.stab_score = min(self.stab_score + 1, STAB_WINDOW)
        else:
            self.stab_score = max(self.stab_score - 1, 0)

        if prev_state:
            self.det_state = (self.stab_score > STAB_OFF)
        else:
            self.det_state = (self.stab_score >= STAB_ON)

        # ---- 전송 로직: "정지 상태" 에지 기반 + 원샷 ----
        stat_prev = self.stationary_state
        self.stationary_state = self._is_stationary() if self.det_state else False

        # (A) 정지 상태로 갓 진입했고, 아직 안 보냈으면 → 딱 1번만 전송
        if self.det_state and self.stationary_state and not stat_prev and not self.stop_sent_latch:
            angle_to_send = float(np.median(self.yaw_hist)) if len(self.yaw_hist) else (angle_val if angle_val is not None else 0.0)
            size_to_send  = self.last_size_str or (size_str if size_str else "general")

            self.stop_sent_latch = True
            self.yaw_baseline = angle_to_send   # ← 정지 순간의 yaw를 기준값으로 저장
            self.get_logger().info(f"[STOP-ONESHOT] send Entry(angle={angle_to_send:.3f} rad) & CarSize(size={size_to_send})")

            if angle_to_send is not None and not self.entry_sent_latch:
                self._send_entry_angle(angle_to_send)
            if size_to_send and not self.carsize_sent:
                self._try_send_carsize(size_to_send)

        # (B) 정지→이동으로 전환되면 래치 해제(다음 정지에 다시 1회 전송)
        elif self.det_state and (not self.stationary_state) and stat_prev:
            self.stop_sent_latch = False

        # (C) 탐지 자체가 꺼지면 전체 리셋
        elif (not self.det_state) and prev_state:
            self.entry_sent_latch = False
            self.carsize_sent = False
            self.stop_sent_latch = False
            self.stationary_state = False
            self.center_buf.clear()
            self.yaw_hist.clear()
            self.img_center_buf.clear()
            self.aabb_buf.clear()
            self.yaw_baseline = None
            self.yaw_for_vis = None

        # 페이드 표시
        if main_pts is not None:
            self.last_main_pts = main_pts.copy()

        alpha = float(self.stab_score) / float(STAB_WINDOW) if STAB_WINDOW > 0 else (1.0 if self.det_state else 0.0)
        alpha = max(0.0, min(1.0, alpha))

        if self.last_main_pts is not None and alpha > 0.0:
            overlay = view.copy()
            cv2.polylines(overlay, [self.last_main_pts], True, (0,255,255), 2)
            cv2.addWeighted(overlay, alpha, view, 1.0 - alpha, 0.0, view)

        # 안정도 막대 & FPS
        bar_w = int(150 * alpha)
        cv2.rectangle(view, (10, 10), (10 + 150, 25), (60,60,60), 1)
        cv2.rectangle(view, (10, 10), (10 + bar_w, 25), (0, 220, 255), -1)
        cv2.putText(view, f"stab:{self.stab_score}/{STAB_WINDOW}", (170, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,255), 1)

        t1 = time.time()
        self.fps = 0.9*self.fps + 0.1*(1.0/max(t1 - self._t_prev, 1e-6))
        self._t_prev = t1
        cv2.putText(view, f"FPS:{self.fps:.1f}", (10,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # 우측 상단 yaw 컴퍼스
        if self.det_state and (self.yaw_for_vis is not None):
            draw_yaw_compass(view, self.yaw_for_vis, yaw0_rad=self.yaw_baseline)


        # 우측 하단 패널: state, size, yaw 표기 (L 제거)
        if panel_values is not None:
            size_v, yaw_v = panel_values
            state_v = "STOP" if self.stationary_state else "MOVING"
            draw_bottom_right_panel(
                view,
                [
                    f"state : {state_v}",
                    f"size : {size_v}",
                    f"yaw : {yaw_v:.3f} rad",
                ]
            )

        cv2.imshow("Detections (annotated)", view if self.show_annot else frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('a'):
            self.show_annot = not self.show_annot
        if k == ord('q'):
            self.get_logger().info("Quit requested")
            rclpy.shutdown()

    # ---------- Cleanup ----------
    def destroy_node(self):
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = ObbEntryCarSizeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()