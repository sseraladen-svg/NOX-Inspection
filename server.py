import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request

from pipeline.router import PipelineRouter
from utils.config_loader import load_config
from utils.overlay import draw_aoi_overlay

app = Flask(__name__)
CONFIG = load_config()
router: PipelineRouter | None = None

DATA_DIR = Path("data")
for directory in ("original", "annotated", "pass", "fail", "logs"):
    (DATA_DIR / directory).mkdir(parents=True, exist_ok=True)

CSV_FILE = DATA_DIR / "logs" / "inspection.csv"
if not CSV_FILE.exists():
    with CSV_FILE.open("w", newline="", encoding="utf-8") as log_file:
        csv.writer(log_file).writerow(
            [
                "ImageID",
                "Timestamp",
                "Result",
                "DefectType",
                "Confidence",
                "Preprocessing(ms)",
                "Detection(ms)",
                "Classification(ms)",
                "Stage 4 Anomaly(ms)",
                "Stage 5 Decision(ms)",
                "Total(ms)",
            ]
        )


@app.route("/")
def index():
    return "BatteryVisionAI Server is Running! Use POST /upload to send images."


@app.route("/upload", methods=["POST"])
def upload():
    if router is None:
        return "Inspection pipeline is not initialized", 503

    image_bytes = request.data
    if len(image_bytes) == 0:
        return "No Image", 400

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return "Invalid Image", 400

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    station_id = request.headers.get("X-Station-ID", "default")
    camera_id = request.headers.get("X-Camera-ID", "default")

    original_path = DATA_DIR / "original" / f"{image_id}.jpg"
    if not cv2.imwrite(str(original_path), frame):
        return "Could not save input image", 500

    decision, timings = router.run(frame, station_id, camera_id)
    processed = decision.pop("_processed_frame")
    annotated_frame = draw_aoi_overlay(processed, decision["detections"])
    annotated_path = DATA_DIR / "annotated" / f"{image_id}.jpg"
    if not cv2.imwrite(str(annotated_path), annotated_frame):
        return "Could not save annotated image", 500

    result = decision["result"]
    defect = decision["defect_type"]
    confidence = decision["confidence"]
    destination = DATA_DIR / result.lower() / f"{image_id}.jpg"
    shutil.copy2(original_path, destination)

    with CSV_FILE.open("a", newline="", encoding="utf-8") as log_file:
        csv.writer(log_file).writerow(
            [
                image_id,
                timestamp,
                result,
                defect,
                confidence,
                round(timings["preprocessing_ms"], 2),
                round(timings["detection_ms"], 2),
                round(timings["classification_ms"], 2),
                round(timings["stage_4_anomaly_ms"], 2),
                round(timings["stage_5_decision_ms"], 2),
                round(timings["total_ms"], 2),
            ]
        )

    print(
        f"[{timestamp}] {result} | Defect: {defect} | "
        f"Preprocessing: {timings['preprocessing_ms']:.2f}ms | "
        f"Detection: {timings['detection_ms']:.2f}ms | "
        f"Classification: {timings['classification_ms']:.2f}ms | "
        f"Stage 4 anomaly: {timings['stage_4_anomaly_ms']:.2f}ms | "
        f"Stage 5 decision: {timings['stage_5_decision_ms']:.2f}ms | "
        f"Total: {timings['total_ms']:.2f}ms"
    )
    return "OK", 200


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Battery cell AOI inspection server")
    parser.add_argument("--product", default="prismatic_cell")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config_path = args.config or Path("config") / f"{args.product}.yaml"
    if config_path.is_file():
        CONFIG = load_config(config_path)
    else:
        print(
            f"WARNING: {config_path} not found; using built-in pipeline defaults."
        )
    router = PipelineRouter(CONFIG)
    app.run(host=args.host, port=args.port, debug=False)
