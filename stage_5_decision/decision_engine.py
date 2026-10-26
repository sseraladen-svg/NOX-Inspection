from __future__ import annotations

from typing import Any


class DecisionEngine:
    """Convert classified detections into an inspection decision."""

    def decide(
        self,
        detections: list[dict[str, Any]],
        station_id: str = "default",
        camera_id: str = "default",
        anomaly_score: float | None = None,
        anomaly_threshold: float | None = None,
    ) -> dict[str, Any]:
        failed_detections = [
            detection
            for detection in detections
            if detection.get("severity", "FAIL") != "PASS"
        ]
        is_anomalous = (
            anomaly_score is not None
            and anomaly_threshold is not None
            and anomaly_score >= anomaly_threshold
        )
        if failed_detections:
            primary = max(
                failed_detections,
                key=lambda detection: float(detection.get("confidence", 0.0)),
            )
            result = "FAIL"
            defect_type = primary.get(
                "defect_type", primary.get("label", primary.get("class", "unknown"))
            )
            confidence = float(primary.get("confidence", 0.0))
        elif is_anomalous:
            result, defect_type, confidence = "FAIL", "anomaly", float(anomaly_score)
        else:
            result, defect_type, confidence = "PASS", "None", 1.0

        return {
            "result": result,
            "defect_type": defect_type,
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "detections": detections,
            "station_id": station_id,
            "camera_id": camera_id,
        }
