#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
import easyocr
import cv2
import re
import time
from collections import Counter
from PIL import ImageFont, ImageDraw, Image   # 한글 출력
import numpy as np   # 한글 출력용

# --- 서비스 ---
from parking_msgs.srv import CarInfo      # DB 저장 서비스
from std_srvs.srv import Trigger          # Trigger 서비스 (네비게이션과 통신)


def preprocess_roi(roi):
    """OCR 전처리"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.medianBlur(resized, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)    
    return morph


def draw_hangul_text(img, text, position, font_size=30, color=(0, 255, 255)):
    """한글 출력 지원 (OpenCV → Pillow 변환 후 출력)"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)


class LicensePlateNode(Node):
    def __init__(self):
        super().__init__('license_plate_node')
        self.bridge = CvBridge()

        # --- PATH ----
        pkg_root = Path(__file__).resolve().parent
        MODEL_PATH = str(pkg_root / "models" / "yolo_license_plate.pt")


        # YOLO + OCR 초기화
        self.model = YOLO(MODEL_PATH)
        self.reader = easyocr.Reader(['ko', 'en'], gpu=True)

        # ✅ 서비스 서버 (네비게이션 → 시작 신호)
        self.srv_start = self.create_service(
            Trigger, '/parking_in/lift_pose_success', self.start_callback
        )

        # ✅ 서비스 클라이언트
        self.cli_db = self.create_client(CarInfo, '/parking_in/car_info_in')
        self.cli_done = self.create_client(Trigger, '/parking_in/car_number')

        # OCR 결과 누적용
        self.ocr_results = []
        self.start_time = None
        self.running = False

        # subscriber 핸들
        self.sub = None

        # ✅ 번호판 후보군 (사용자 지정)
        self.candidate_list = ["381수8717", "363소6691", "144도4628", "306고5868", "276어9461"]

        self.get_logger().info("📷 번호판 인식 노드 준비 완료 (/parking_in/lift_pose_success 대기중)")

    # ✅ 네비게이션에서 시작 트리거 받으면 동작 시작
    def start_callback(self, request, response):
        self.get_logger().info("📥 네비게이션 → lift_pose_success 수신 → 번호판 인식 시작")

        if self.sub is None:
            self.sub = self.create_subscription(
                CompressedImage,
                '/robot3/oakd/rgb/image_raw/compressed',
                self.image_callback,
                10
            )

        self.running = True
        self.start_time = time.time()
        self.ocr_results.clear()

        response.success = True
        response.message = "번호판 인식 시작됨"
        return response

    def image_callback(self, msg):
        if not self.running:
            return

        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        results = self.model(frame,imgsz = 960, verbose=False) #의미가 있는 부분이면 넣고 아니면 잘라내자
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf < 0.5 :
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 번호판 ROI 추출
                roi = frame[y1:y2, x1:x2]
                roi_processed = preprocess_roi(roi)

                # OCR 실행
                ocr_result = self.reader.readtext(
                    roi_processed, detail=0, paragraph=False,
                    contrast_ths=0.05, adjust_contrast=0.7,
                    text_threshold=0.4, decoder='beamsearch'
                )
                if ocr_result:
                    for token in ocr_result:
                        cleaned = re.sub(r"[\s\(\)\[\]\{\}]", "", token)
                        match = re.findall(r"[0-9]{2,3}[가-힣][0-9]{4}", cleaned)
                        if match:
                            plate = match[0]
                            self.ocr_results.append(plate)
                            frame = draw_hangul_text(frame, plate, (x1, y1-30))

        cv2.imshow("License Plate Detection", frame)
        cv2.waitKey(1)

        # 5초마다 결과 집계
        elapsed = time.time() - self.start_time
        if elapsed >= 5.0:
            self.finish_process()

    def finish_process(self):
        """5초간 결과 집계 후 후보군 비교 + DB 저장 + 네비게이션 알림"""
        if self.ocr_results:
            counter = Counter(self.ocr_results)
            plate_text, _ = counter.most_common(1)[0]
            self.get_logger().info(f"📊 최빈 번호판 후보: {plate_text}")

            # ✅ 후보군 리스트 안에 있는 경우만 성공 처리
            if plate_text in self.candidate_list:
                self.get_logger().info(f"✅ 후보군과 일치: {plate_text}")

                # DB 서비스 연결 확인
                if self.cli_db.wait_for_service(timeout_sec=2.0):
                    req = CarInfo.Request()
                    req.license_plate = plate_text
                    self.cli_db.call_async(req)
                    self.get_logger().info(f"📤 DB 서비스 호출: 번호판 = {plate_text}")

                    # DB 서비스 연결에 성공했으므로 Trigger 전송
                    if self.cli_done.wait_for_service(timeout_sec=2.0):
                        req2 = Trigger.Request()
                        self.cli_done.call_async(req2)
                        self.get_logger().info("📤 네비게이션에 완료 Trigger(/parking_in/car_number) 전송")

                    # 카메라 구독 해제 (성공 시 종료)
                    if self.sub:
                        self.destroy_subscription(self.sub)
                        self.sub = None
                    cv2.destroyAllWindows()
                    self.running = False
                    return
                else:
                    self.get_logger().warn("❌ DB 서비스 연결 실패 → 5초 재시도")
            else:
                self.get_logger().info(f"❌ {plate_text} 는 후보군에 없음 → 5초 재시도")
        else:
            self.get_logger().info("❌ 5초 동안 번호판 인식 실패 → 5초 재시도")

        # 실패 → 재시도
        self.start_time = time.time()
        self.ocr_results.clear()


def main(args=None):
    rclpy.init(args=args)
    node = LicensePlateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()