"""
core/overlay.py

Fixes the schema mismatch in the original aoi_overlay.py: that file expected
d["bbox"], d["severity"], d["class"] but core/detector.py produced
d["label"], d["confidence"], d["mask"] with no bbox/severity — so it was
dead code, never callable end to end.

Now consumes detections as annotated by core/decision_engine.py:
{"label", "class_id", "confidence", "bbox", "mask", "severity"}.
"""

import cv2

SEVERITY_COLORS = {
    "FAIL": (0, 0, 255),    # red
    "WARN": (0, 255, 255),  # yellow
    "NONE": (0, 255, 0),    # green
}


def get_color(severity: str):
    return SEVERITY_COLORS.get(severity, (0, 255, 0))


def draw_aoi_overlay(frame, detections):
    """
    detections: list of dicts with "bbox", "severity", "label", "confidence"
    (i.e. the output of DecisionEngine.evaluate_detection / evaluate()["detections"]).
    """
    overlay = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = get_color(d["severity"])

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        cv2.putText(
            overlay,
            f'{d["label"]}:{d["confidence"]:.2f}',
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    return frame
