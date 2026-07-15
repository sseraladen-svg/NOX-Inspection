from ultralytics import YOLO

def main():
    print("BatteryVisionAI Training Started\n")
    
    model = YOLO("models/yolo11n-seg.pt")

    model.train(
        data="dataset/data.yaml", # Ensure this file exists in Colab
        epochs=50, # Reduced for Colab free tier testing
        imgsz=640,
        batch=8,
        workers=2, # Colab has limited workers
        device=0,  # Uses Colab's GPU
        project="runs/train",
        name="BatteryVisionAI",
        pretrained=True,
        patience=10,
        plots=True
    )

    print("Training Completed")
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()