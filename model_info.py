from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n.pt")

# Print architecture
print(model.model)