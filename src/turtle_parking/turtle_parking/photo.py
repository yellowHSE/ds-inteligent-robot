# grab_frame.py
import cv2, time
DEV = "/dev/video2"   # 웹캠이면 이대로, 파일이면 "video.mp4", ROS라면 이미지 저장 후 사용
W,H,FPS = 640,640,30

cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS,          FPS)

ok, frame = cap.read()
if not ok:
    raise SystemExit("카메라 프레임 획득 실패")

cv2.imshow("frame", frame)
cv2.imwrite("frame_for_calib.jpg", frame)
print("Saved: frame_for_calib.jpg")
cv2.waitKey(500)
cap.release(); cv2.destroyAllWindows()
