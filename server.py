import sys, os
sys.path.append(os.path.abspath('.')) # CRITICAL FIX FOR COLAB

from flask import Flask, request
import cv2, numpy as np, csv, shutil, time
from datetime import datetime
from detection.detector import Detector
from aoi_overlay import draw_aoi_overlay

app = Flask(__name__)

@app.route("/")
def index():
    return "BatteryVisionAI Server is Running! Use POST /upload to send images."

DATA_DIR = "data"
for d in ["original", "annotated", "pass", "fail", "logs"]:
    os.makedirs(os.path.join(DATA_DIR, d), exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "logs", "inspection.csv")
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["ImageID", "Timestamp", "Result", "DefectType", "Confidence", "Time(ms)"])

detector = Detector(seg_model_path="models/yolo11n-seg.pt", conf=0.5)

@app.route("/upload", methods=["POST"])
def upload():
    image_bytes = request.data
    if len(image_bytes) == 0: return "No Image", 400

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None: return "Invalid Image", 400

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    original_path = os.path.join(DATA_DIR, "original", f"{image_id}.jpg")
    cv2.imwrite(original_path, frame)

    roi = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    start = time.perf_counter()
    annotated_frame, detections = detector.detect(roi)
    annotated_frame = draw_aoi_overlay(annotated_frame, detections)
    annotated_frame = detector.draw_info(annotated_frame, detections)
    end = time.perf_counter()
    
    inference_time = round((end - start) * 1000, 2)
    annotated_path = os.path.join(DATA_DIR, "annotated", f"{image_id}.jpg")
    cv2.imwrite(annotated_path, annotated_frame)

    if len(detections) > 0:
        result, defect, confidence = "FAIL", detections[0]["label"], detections[0]["confidence"]
        shutil.copy(original_path, os.path.join(DATA_DIR, "fail", f"{image_id}.jpg"))
    else:
        result, defect, confidence = "PASS", "None", 1.00
        shutil.copy(original_path, os.path.join(DATA_DIR, "pass", f"{image_id}.jpg"))

    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([image_id, timestamp, result, defect, confidence, inference_time])

    print(f"[{timestamp}] {result} | Defect: {defect} | Time: {inference_time}ms")
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)