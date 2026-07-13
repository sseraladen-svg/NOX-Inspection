from flask import Flask, request
import cv2
import numpy as np
import os
import csv
import shutil
import time
from datetime import datetime

from detection.detector import Detector

app = Flask(__name__)

# ==========================================================
# Folder Structure
# ==========================================================
DATA_DIR = "data"

ORIGINAL_DIR = os.path.join(DATA_DIR, "original")
ANNOTATED_DIR = os.path.join(DATA_DIR, "annotated")
PASS_DIR = os.path.join(DATA_DIR, "pass")
FAIL_DIR = os.path.join(DATA_DIR, "fail")
LOG_DIR = os.path.join(DATA_DIR, "logs")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(PASS_DIR, exist_ok=True)
os.makedirs(FAIL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
# CSV Log File
# ==========================================================
CSV_FILE = os.path.join(LOG_DIR, "inspection.csv")

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ImageID",
            "Timestamp",
            "CameraID",
            "Result",
            "DefectType",
            "Confidence",
            "InferenceTime(ms)",
            "OriginalImage",
            "AnnotatedImage"
        ])

# ==========================================================
# Load YOLO Model
# ==========================================================
detector = Detector(
    seg_model_path="models/yolo11n-seg.pt",
    conf=0.5
)

print("[BatteryVisionAI] YOLO Model Loaded")

CAMERA_ID = "ESP32_CAM_01"

# ==========================================================
# Upload Route
# ==========================================================
@app.route("/upload", methods=["POST"])
def upload():

    image_bytes = request.data

    if len(image_bytes) == 0:
        return "No Image", 400

    # ------------------------------------------------------
    # Decode Image
    # ------------------------------------------------------
    nparr = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return "Invalid Image", 400

    # ------------------------------------------------------
    # Image ID + Timestamp
    # ------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    image_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # ------------------------------------------------------
    # Save Original Image
    # ------------------------------------------------------
    original_path = os.path.join(
        ORIGINAL_DIR,
        f"{image_id}.jpg"
    )

    cv2.imwrite(original_path, frame)

    # ------------------------------------------------------
    # ROI (Adjust Coordinates Later)
    # ------------------------------------------------------
    roi = frame

    # Example:
    # roi = frame[100:500, 150:650]

    # ------------------------------------------------------
    # Resize
    # ------------------------------------------------------
    roi = cv2.resize(
        roi,
        (640, 640),
        interpolation=cv2.INTER_LINEAR
    )

    # ------------------------------------------------------
    # Noise Reduction
    # ------------------------------------------------------
    roi = cv2.GaussianBlur(
        roi,
        (3, 3),
        0
    )

    # ------------------------------------------------------
    # YOLO Inference
    # ------------------------------------------------------
    start = time.perf_counter()

    annotated_frame, detections = detector.detect(roi)

    annotated_frame = detector.draw_info(
        annotated_frame,
        detections
    )

    end = time.perf_counter()

    inference_time = round(
        (end - start) * 1000,
        2
    )

    # ------------------------------------------------------
    # Save Annotated Image
    # ------------------------------------------------------
    annotated_path = os.path.join(
        ANNOTATED_DIR,
        f"{image_id}.jpg"
    )

    cv2.imwrite(
        annotated_path,
        annotated_frame
    )

    # ------------------------------------------------------
    # PASS / FAIL
    # ------------------------------------------------------
    if len(detections) > 0:

        result = "FAIL"

        defect = detections[0]["label"]

        confidence = detections[0]["confidence"]

        shutil.copy(
            original_path,
            os.path.join(
                FAIL_DIR,
                f"{image_id}.jpg"
            )
        )

    else:

        result = "PASS"

        defect = "None"

        confidence = 1.00

        shutil.copy(
            original_path,
            os.path.join(
                PASS_DIR,
                f"{image_id}.jpg"
            )
        )

    # ------------------------------------------------------
    # CSV Logging
    # ------------------------------------------------------
    with open(
        CSV_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            image_id,
            timestamp,
            CAMERA_ID,
            result,
            defect,
            confidence,
            inference_time,
            original_path,
            annotated_path
        ])

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------
    print("\n====================================")
    print(f"Image ID       : {image_id}")
    print(f"Timestamp      : {timestamp}")
    print(f"Result         : {result}")
    print(f"Defect         : {defect}")
    print(f"Confidence     : {confidence}")
    print(f"Inference Time : {inference_time} ms")
    print("====================================")

    return "OK", 200


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    print("\n====================================")
    print(" BatteryVisionAI Server Started")
    print(" Waiting for ESP32-CAM Images...")
    print(" Upload URL:")
    print(" http://0.0.0.0:5000/upload")
    print("====================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )