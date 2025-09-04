# calibrate_homography_click.py
import cv2, yaml, numpy as np

IMG_PATH = "frame_for_calib.jpg"   # 캘리브용 이미지 경로
# 바닥의 실세계 크기(cm). '직사각형' 기준(좌상→우상→좌하→우하)일 때 사용
WIDTH_CM  = 120    # 가로(좌→우)
HEIGHT_CM = 70.0    # 세로(위→아래)

# 출력 해상도 스케일: 1cm를 몇 픽셀로 볼지 (미리보기·warp용)
PX_PER_CM = 2.0

clicked = []
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append([x, y]); print(f"clicked {len(clicked)}:", (x, y))

img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit(f"이미지를 열 수 없습니다: {IMG_PATH}")

print("바닥 네 점을 '좌상→우상→좌하→우하' 순서로 클릭하세요. (ESC 종료)")
cv2.namedWindow("calib", cv2.WINDOW_NORMAL); cv2.resizeWindow("calib", 960, 540)
cv2.setMouseCallback("calib", on_mouse)

while True:
    disp = img.copy()
    for i, p in enumerate(clicked):
        cv2.circle(disp, tuple(map(int,p)), 7, (0,255,0), -1)
        cv2.putText(disp, str(i+1), (int(p[0])+6, int(p[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("calib", disp)
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
if len(clicked) != 4:
    raise SystemExit("4점을 모두 찍지 않았습니다. 스크립트를 다시 실행하세요.")

src = np.array(clicked, dtype=np.float32)

# 목적지 좌표(센티미터 단위 평면). 좌상(0,0)→우상(WIDTH,0)→좌하(0,HEIGHT)→우하(WIDTH,HEIGHT)
dst_cm = np.array([[0,0],[WIDTH_CM,0],[0,HEIGHT_CM],[WIDTH_CM,HEIGHT_CM]], dtype=np.float32)

# 호모그래피 계산 (px → cm)
H, mask = cv2.findHomography(src, dst_cm, method=cv2.RANSAC)
if H is None:
    raise SystemExit("호모그래피 계산 실패")

# 저장
data = {
    "H": H.tolist(),                # 픽셀 → 센티미터 변환 행렬(3x3)
    "width_cm": float(WIDTH_CM),
    "height_cm": float(HEIGHT_CM),
    "px_per_cm": float(PX_PER_CM),  # 미리보기용 스케일
    "image_size": {"w": int(img.shape[1]), "h": int(img.shape[0])},
    "click_order": "TL, TR, BL, BR"
}
with open("homography.yaml", "w") as f:
    yaml.safe_dump(data, f)
print("Saved: homography.yaml")

# 간단 미리보기 (바로 확인)
W = int(WIDTH_CM * PX_PER_CM)
Hh = int(HEIGHT_CM * PX_PER_CM)
bev = cv2.warpPerspective(img, H * PX_PER_CM, (W, Hh))  # 스케일 포함
# 50cm 그리드
for y in range(0, Hh, int(50*PX_PER_CM)):
    cv2.line(bev, (0,y), (W,y), (0,255,0), 1)
for x in range(0, W, int(50*PX_PER_CM)):
    cv2.line(bev, (x,0), (x,Hh), (0,255,0), 1)
cv2.imshow("BEV preview (50cm grid)", bev)
cv2.waitKey(0); cv2.destroyAllWindows()
