"""
core/decision_engine.py

Separates AI perception (detector/classifier/anomaly outputs) from
manufacturing PASS/FAIL logic, as claimed in the pitch deck's architecture.
All thresholds/severity come from the product config — swapping products
means swapping config, not this file.

This is the single place PASS/FAIL is decided. server.py and main.py must
call this instead of deciding PASS/FAIL themselves.
"""


class DecisionEngine:
    def __init__(self, config: dict):
        rules = config["decision_rules"]
        self.fail_confidence_threshold = rules["fail_confidence_threshold"]
        self.severity_map = rules["severity_map"]

    def _severity_for(self, label: str) -> str:
        return self.severity_map.get(label, "FAIL")  # unknown class defaults to FAIL, fail-safe

    def evaluate_detection(self, detection: dict) -> dict:
        """
        Attach severity to a single detection dict from core/detector.py.
        Returns the detection dict with a "severity" key added.
        """
        detection["severity"] = self._severity_for(detection["label"])
        return detection

    def evaluate(self, detections: list, station_id: str = None, camera_id: str = None) -> dict:
        """
        Evaluate a full set of detections for one image/camera capture.

        Returns:
            {
                "result": "PASS" | "FAIL",
                "defect": str,          # top defect label, "None" if PASS
                "confidence": float,
                "severity": str,        # "WARN" | "FAIL" | "NONE"
                "station_id": str | None,
                "camera_id": str | None,
                "detections": list,     # each detection annotated with severity
            }
        """
        if not detections:
            return {
                "result": "PASS",
                "defect": "None",
                "confidence": 1.00,
                "severity": "NONE",
                "station_id": station_id,
                "camera_id": camera_id,
                "detections": [],
            }

        annotated = [self.evaluate_detection(d) for d in detections]

        # A single FAIL-severity or above-threshold detection fails the whole capture.
        fail_hits = [
            d for d in annotated
            if d["severity"] == "FAIL" and d["confidence"] >= self.fail_confidence_threshold
        ]

        if fail_hits:
            worst = max(fail_hits, key=lambda d: d["confidence"])
            result = "FAIL"
            defect = worst["label"]
            confidence = worst["confidence"]
            severity = "FAIL"
        else:
            # only WARN-severity detections present
            worst = max(annotated, key=lambda d: d["confidence"])
            result = "FAIL" if worst["confidence"] >= self.fail_confidence_threshold else "PASS"
            defect = worst["label"] if result == "FAIL" else "None"
            confidence = worst["confidence"] if result == "FAIL" else 1.00
            severity = worst["severity"] if result == "FAIL" else "NONE"

        return {
            "result": result,
            "defect": defect,
            "confidence": confidence,
            "severity": severity,
            "station_id": station_id,
            "camera_id": camera_id,
            "detections": annotated,
        }
