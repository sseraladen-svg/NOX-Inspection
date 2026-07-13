import cv2
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, seg_model_path="models/yolo11n-seg.pt", conf=0.5):
        self.model = YOLO(seg_model_path)
        self.conf = conf
        print(f"[Detector] Loaded: {seg_model_path}")

    def detect(self, frame):
        """
        Run segmentation on frame.
        Returns: annotated frame + list of detections
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []

        for result in results:
            # Draw segmentation masks
            annotated = result.plot()

            if result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    # Class info
                    cls_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    label = self.model.names[cls_id]

                    detections.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "mask": mask.cpu().numpy()
                    })

            return annotated, detections

        return frame, []

    def draw_info(self, frame, detections):
        """Draw detection count and labels on frame."""
        h, w = frame.shape[:2]

        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

        count = len(detections)
        status = f"Detections: {count}"
        color = (0, 255, 0) if count == 0 else (0, 165, 255)

        cv2.putText(frame, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # List each detection
        for i, d in enumerate(detections):
            text = f"{d['label']} ({d['confidence']})"
            cv2.putText(frame, text, (10, 70 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        return frame