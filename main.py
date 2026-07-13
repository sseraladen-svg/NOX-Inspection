import cv2
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection.detector import Detector

IMAGE_PATH = "received_images/latest.jpg"

def main():

    print("[BatteryVisionAI] Starting...")

    detector = Detector(
        seg_model_path="models/yolo11n-seg.pt",
        conf=0.5
    )

    last_modified = 0

    while True:

        if not os.path.exists(IMAGE_PATH):
            time.sleep(0.5)
            continue

        modified = os.path.getmtime(IMAGE_PATH)

        if modified != last_modified:

            last_modified = modified

            frame = cv2.imread(IMAGE_PATH)

            if frame is None:
                continue

            annotated_frame, detections = detector.detect(frame)
            annotated_frame = detector.draw_info(annotated_frame, detections)

            cv2.imshow("BatteryVisionAI", annotated_frame)

            if detections:
                print("\nDetections:")
                for d in detections:
                    print(f"{d['label']} | {d['confidence']:.2f}")
            else:
                print("PASS")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()