import cv2
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, seg_model_path="models/yolo11n-seg.pt", conf=0.5, cnn_model=None):
        self.model = YOLO(seg_model_path)
        self.conf = conf
        self.cnn_model = cnn_model
        print(f"[Detector] Loaded: {seg_model_path}")

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []
        for result in results:
            annotated = result.plot()
            if result.masks is not None and result.boxes is not None:
                for i, mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    label = self.model.names[cls_id]
                    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].tolist())
                    
                    if self.cnn_model is not None:
                        crop = frame[y1:y2, x1:x2]
                        label = self.cnn_model.predict(crop) 

                    detections.append({
                        "label": label, "class": label, "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2], "severity": "FAIL", "mask": mask.cpu().numpy()
                    })
        return annotated, detections

    def draw_info(self, frame, detections):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        count = len(detections)
        status = f"Detections: {count}"
        color = (0, 255, 0) if count == 0 else (0, 165, 255)
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        for i, d in enumerate(detections):
            text = f"{d['label']} ({d['confidence']})"
            cv2.putText(frame, text, (10, 70 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        return frame