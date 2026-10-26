from __future__ import annotations

import time
from typing import Any

from stage_1_preprocessing.preprocessor import Preprocessor
from stage_2_detection.detector import Detector
from stage_3_classification.classifier import Classifier
from stage_4_anomaly.anomaly_detector import AnomalyDetector
from stage_5_decision.decision_engine import DecisionEngine


class PipelineRouter:
    """Initialize the inspection stages once and route each frame through them."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        preprocessing_config = config.get("preprocessing", {})
        detection_config = config.get("detection", {})
        classification_config = config.get("classification", {})
        anomaly_config = config.get("anomaly_detection", {})

        self.preprocessor = Preprocessor(preprocessing_config)
        self.detector = Detector(
            seg_model_path=detection_config.get("model_path", "models/yolo11n-seg.pt"),
            conf=float(detection_config.get("confidence", 0.5)),
        )
        self.classifier = Classifier(
            model_path=classification_config.get(
                "model_path", "models/classifier.onnx"
            ),
            class_names=classification_config.get(
                "class_names", ["scratch", "dent", "discoloration"]
            ),
            input_size=int(classification_config.get("input_size", 224)),
        )
        self.anomaly_detector = AnomalyDetector(
            memory_bank_path=anomaly_config.get(
                "memory_bank_path", "models/patchcore_model"
            )
        )
        self.anomaly_threshold = float(anomaly_config.get("threshold", 0.5))
        self.decision_engine = DecisionEngine()

    def run(
        self, frame: Any, station_id: str, camera_id: str
    ) -> tuple[dict[str, Any], dict[str, float]]:
        total_start = time.perf_counter()

        start = time.perf_counter()
        processed = self.preprocessor.process(frame)
        preprocessing_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        _, detections = self.detector.detect(processed)
        detection_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        if detections:
            classifications = self.classifier.classify_all(processed, detections)
            for detection, classification in zip(detections, classifications):
                detection["detection_confidence"] = detection.get("confidence", 0.0)
                detection["defect_type"] = classification["defect_type"]
                detection["classification_confidence"] = classification["confidence"]
                detection["confidence"] = classification["confidence"]
                detection["label"] = classification["defect_type"]
                detection["class"] = classification["defect_type"]
        classification_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        anomaly_score = self.anomaly_detector.predict(processed)
        anomaly_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        decision = self.decision_engine.decide(
            detections,
            station_id=station_id,
            camera_id=camera_id,
            anomaly_score=anomaly_score,
            anomaly_threshold=self.anomaly_threshold,
        )
        decision["_processed_frame"] = processed
        decision_ms = (time.perf_counter() - start) * 1000

        timings = {
            "preprocessing_ms": preprocessing_ms,
            "detection_ms": detection_ms,
            "classification_ms": classification_ms,
            "stage_4_anomaly_ms": anomaly_ms,
            "stage_5_decision_ms": decision_ms,
            "total_ms": (time.perf_counter() - total_start) * 1000,
        }
        return decision, timings
