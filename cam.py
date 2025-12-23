import cv2
from ultralytics import YOLO
import math
import time

# ================= CONFIG =================
VIDEO_PATH = "video.mp4"
CONF_THRESH = 0.5
PIXEL_TO_METER = 0.02

# ================= MODEL =================
model = YOLO("yolov8s.pt")

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 25

WAIT_TIME = int(1000 / video_fps)

print(f"[INFO] FPS locked at {video_fps}, waitKey = {WAIT_TIME} ms")

# ================= MOUSE =================
mouse_x, mouse_y = 0, 0

def mouse_event(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Detection", mouse_event)

# ================= SIMPLE TRACKING =================
prev_centers = {}   # id -> (x, y, time)
next_id = 1

def assign_id(cx, cy):
    global next_id
    for oid, (px, py, _) in prev_centers.items():
        if abs(cx - px) < 40 and abs(cy - py) < 40:
            return oid
    oid = next_id
    next_id += 1
    return oid

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESH, verbose=False)

    current_centers = {}
    hover_text = None
    now = time.time()
    crowd_count = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        oid = assign_id(cx, cy)

        speed = 0.0
        action = ""

        if oid in prev_centers:
            px, py, pt = prev_centers[oid]
            dist_pixels = math.hypot(cx - px, cy - py)
            dist_meters = dist_pixels * PIXEL_TO_METER
            time_diff = max(now - pt, 0.001)
            speed = dist_meters / time_diff
            action = "Walking" if speed > 0.4 else "Standing"

        current_centers[oid] = (cx, cy, now)

        if label == "person":
            crowd_count += 1

        # ---------- DRAW ----------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} #{oid}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
            if label == "person":
                hover_text = f"Person is {action} | Speed: {speed:.2f} m/s"
            else:
                hover_text = f"This is a {label}"

    # ---------- CROWD COUNT ----------
    h, w, _ = frame.shape
    cv2.putText(
        frame,
        f"People Count: {crowd_count}",
        (w - 320, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3
    )

    if hover_text:
        cv2.putText(
            frame,
            hover_text,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3
        )

    cv2.imshow("Detection", frame)
    prev_centers = current_centers.copy()

    if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
