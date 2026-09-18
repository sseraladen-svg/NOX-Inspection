"""
core/detector.py

Config-driven YOLO segmentation wrapper. Model path and confidence threshold
come from the product config (config/schema.yaml) — nothing hardcoded.

Fixes the schema gap in the original detection/detector.py: each detection
now includes "bbox" (needed by core/overlay.py and core/decision_engine.py),
in addition to "label", "confidence", and "mask".
"""

from ultralytics import YOLO


class Detector:
    def __init__(self, config: dict):
        """
        config: full loaded product config dict.
        Reads config["models"]["yolo_seg"] and config["decision_rules"]["fail_confidence_threshold"].
        """
        model_path = config["models"]["yolo_seg"]
        self.conf = config["decision_rules"]["fail_confidence_threshold"]
        self.class_names = config["defect_classes"]

        self.model = YOLO(model_path)
        print(f"[Detector] Loaded: {model_path} (conf={self.conf})")

    def detect(self, frame):
        """
        Run segmentation on a single frame.
        Returns: (annotated_frame, detections)

        Each detection dict:
            {
                "label": str,          # class name from config.defect_classes
                "class_id": int,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],   # int pixel coords
                "mask": np.ndarray,
            }
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []

        for result in results:
            annotated = result.plot()

            if result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    bbox = result.boxes.xyxy[i].cpu().numpy().astype(int).tolist()

                    label = self.class_names.get(cls_id, self.model.names.get(cls_id, str(cls_id)))

                    detections.append({
                        "label": label,
                        "class_id": cls_id,
                        "confidence": round(conf, 2),
                        "bbox": bbox,
                        "mask": mask.cpu().numpy(),
                    })

            return annotated, detections

        return frame, []
