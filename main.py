import sys
import os
sys.path.append(os.path.abspath('.')) # This fixes the Colab import error
import cv2
import os
from google.colab.patches import cv2_imshow
from detection.detector import Detector
from aoi_overlay import draw_aoi_overlay

# Use a dummy image for testing in Colab
TEST_IMAGE_PATH = "test_battery.jpg" 

def main():
    print("[BatteryVisionAI] Starting Colab Workflow...")
    
    # Create a dummy test image if it doesn't exist
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Creating dummy {TEST_IMAGE_PATH} for testing...")
        dummy = cv2.resize(cv2.imread("models/yolo11n-seg.pt", cv2.IMREAD_UNCHANGED) if os.path.exists("models/yolo11n-seg.pt") else np.zeros((640,640,3), dtype=np.uint8), (640,640))
        cv2.imwrite(TEST_IMAGE_PATH, dummy)

    detector = Detector(seg_model_path="models/yolo11n-seg.pt", conf=0.5)
    
    frame = cv2.imread(TEST_IMAGE_PATH)
    if frame is None:
        print("Error: Could not read test image.")
        return

    # Run Detection
    annotated_frame, detections = detector.detect(frame)
    
    # Apply Overlays
    annotated_frame = draw_aoi_overlay(annotated_frame, detections)
    annotated_frame = detector.draw_info(annotated_frame, detections)

    # Display in Colab
    cv2_imshow(annotated_frame)
    
    if detections:
        print(f"\nDetected {len(detections)} defects:")
        for d in detections:
            print(f"- {d['label']} (Conf: {d['confidence']})")
    else:
        print("\nResult: PASS (No defects found)")

if __name__ == "__main__":
    import numpy as np
    main()