import cv2
from ultralytics import YOLO
from datetime import datetime
import csv
import time

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_PATH = "yolov8n.pt"
DEVICE = "mps"
FRAME_SIZE = (640, 480)
CONF_THRESHOLD = 0.6
FRAME_SKIP = 2
PERSISTENCE = 5  # frames before logging object
OBJECTS = ["cell phone", "wallet", "bottle", "laptop",
           "backpack", "keyboard", "mouse", "book",
           "handbag", "cup"]
CSV_FILE = "detections.csv"

# -------------------------
# CSV SETUP
# -------------------------
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Object", "Duration_Seconds"])

# -------------------------
# LOAD MODEL
# -------------------------
model = YOLO(MODEL_PATH)

# -------------------------
# VIDEO CAPTURE
# -------------------------
cap = cv2.VideoCapture(0)
frame_number = 0
# {object_name: {'first_seen': int, 'last_seen': int, 'frames': list}}
tracked_objects = {}

# -------------------------
# UTILITY FUNCTIONS
# -------------------------


def draw_box(frame, x1, y1, x2, y2, color=(0, 255, 255), thickness=2, size=25):
    """Draw corner-only box"""
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + size, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + size), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - size, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + size), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + size, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - size), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - size, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - size), color, thickness)


def draw_label(frame, text, x, y):
    """Draw semi-transparent label"""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - h - 5), (x + w, y + 5), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    return frame


# -------------------------
# MAIN LOOP
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    if frame_number % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, FRAME_SIZE)
    results = model(frame, stream=True, device=DEVICE)

    detected = []

    # Process detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if label in OBJECTS and conf > CONF_THRESHOLD:
                detected.append(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_box(frame, x1, y1, x2, y2)
                frame = draw_label(frame, label.upper(), x1, y1 - 10)

                # Track object frames
                if label not in tracked_objects:
                    tracked_objects[label] = {'first_seen': frame_number,
                                              'last_seen': frame_number,
                                              'frames': [frame_number]}
                else:
                    tracked_objects[label]['last_seen'] = frame_number
                    tracked_objects[label]['frames'].append(frame_number)

    # Check for objects to log
    to_remove = []
    for obj, data in tracked_objects.items():
        if obj not in detected:
            if frame_number - data['last_seen'] > PERSISTENCE:
                duration = len(data['frames']) / 30  # approximate seconds
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now(), obj, round(duration, 2)])
                to_remove.append(obj)

    for obj in to_remove:
        del tracked_objects[obj]

    # Display the frame
    cv2.imshow("Smart Scanner", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
