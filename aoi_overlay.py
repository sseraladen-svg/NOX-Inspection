import cv2

def get_color(severity):
    if severity == "FAIL": return (0, 0, 255)
    elif severity == "WARN": return (0, 255, 255)
    else: return (0, 255, 0)

def draw_aoi_overlay(frame, detections):
    overlay = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = get_color(d["severity"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.putText(overlay, f'{d["class"]}:{d["confidence"]:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    return frame