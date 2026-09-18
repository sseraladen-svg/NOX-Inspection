import argparse
from flask import Flask, request
import cv2
import numpy as np
import os
import csv
import shutil
import time
from datetime import datetime

from core.config_loader import load_config
from core.preprocessing import Preprocessor
from core.detector import Detector
from core.decision_engine import DecisionEngine
from core.overlay import draw_aoi_overlay

# ==========================================================
# CLI / Config
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--product", default="prismatic_cell", help="Product config id under config/")
args, _ = parser.parse_known_args()

CONFIG = load_config(args.product)

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
# CSV Log File — now includes per-camera/per-station traceability
# ==========================================================
CSV_FILE = os.path.join(LOG_DIR, "inspection.csv")

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ImageID", "Timestamp", "ProductID", "StationID", "CameraID",
            "Result", "Severity", "DefectType", "Confidence",
            "InferenceTime(ms)", "OriginalImage", "AnnotatedImage"
        ])

# ==========================================================
# Core pipeline modules — all config-driven
# ==========================================================
preprocessor = Preprocessor(CONFIG)
detector = Detector(CONFIG)
decision_engine = DecisionEngine(CONFIG)

print(f"[NOX Inspection] Loaded product config: {CONFIG['product_id']}")

DEFAULT_STATION_ID = "station_01"
DEFAULT_CAMERA_ID = "cam_01"

# ==========================================================
# Upload Route
# ==========================================================
@app.route("/upload", methods=["POST"])
def upload():

    image_bytes = request.data

    if len(image_bytes) == 0:
        return "No Image", 400

    station_id = request.args.get("station_id", DEFAULT_STATION_ID)
    camera_id = request.args.get("camera_id", DEFAULT_CAMERA_ID)

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
    # Preprocessing (OpenCV stage, config-driven: ROI/resize/denoise/CLAHE)
    # ------------------------------------------------------
    processed = preprocessor.process(frame)

    # ------------------------------------------------------
    # YOLO Inference
    # ------------------------------------------------------
    start = time.perf_counter()

    _, detections = detector.detect(processed)

    # ------------------------------------------------------
    # Decision Engine — PASS/FAIL decided here, not inline
    # ------------------------------------------------------
    decision = decision_engine.evaluate(detections, station_id=station_id, camera_id=camera_id)

    annotated_frame = draw_aoi_overlay(processed, decision["detections"])

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
    # PASS / FAIL routing
    # ------------------------------------------------------
    dest_dir = FAIL_DIR if decision["result"] == "FAIL" else PASS_DIR
    shutil.copy(original_path, os.path.join(dest_dir, f"{image_id}.jpg"))

    # ------------------------------------------------------
    # CSV Logging — per-camera/per-station traceability
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
            CONFIG["product_id"],
            station_id,
            camera_id,
            decision["result"],
            decision["severity"],
            decision["defect"],
            decision["confidence"],
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
    print(f"Station/Camera : {station_id} / {camera_id}")
    print(f"Result         : {decision['result']} ({decision['severity']})")
    print(f"Defect         : {decision['defect']}")
    print(f"Confidence     : {decision['confidence']}")
    print(f"Inference Time : {inference_time} ms")
    print("====================================")

    return decision["result"], 200


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    print("\n====================================")
    print(f" NOX Inspection Server — product: {CONFIG['product_id']}")
    print(" Waiting for camera images...")
    print(" Upload URL:")
    print(" http://0.0.0.0:5000/upload")
    print("====================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )