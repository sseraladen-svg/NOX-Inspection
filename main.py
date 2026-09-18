import cv2
import time
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config_loader import load_config
from core.preprocessing import Preprocessor
from core.detector import Detector
from core.decision_engine import DecisionEngine
from core.overlay import draw_aoi_overlay

IMAGE_PATH = "received_images/latest.jpg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", default="prismatic_cell", help="Product config id under config/")
    args = parser.parse_args()

    config = load_config(args.product)

    print(f"[NOX Inspection] Starting... (product={config['product_id']})")

    preprocessor = Preprocessor(config)
    detector = Detector(config)
    decision_engine = DecisionEngine(config)

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

            processed = preprocessor.process(frame)
            _, detections = detector.detect(processed)
            decision = decision_engine.evaluate(detections)
            annotated_frame = draw_aoi_overlay(processed, decision["detections"])

            cv2.imshow("NOX Inspection", annotated_frame)

            print(f"\nResult: {decision['result']} ({decision['severity']})")
            if decision["detections"]:
                for d in decision["detections"]:
                    print(f"  {d['label']} | {d['confidence']:.2f} | {d['severity']}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
