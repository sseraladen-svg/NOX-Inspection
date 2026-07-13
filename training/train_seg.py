from ultralytics import YOLO


def main():

    print("===================================")
    print(" BatteryVisionAI Training Started")
    print("===================================\n")

    # Load pretrained YOLO11 Segmentation model
    model = YOLO("models/yolo11n-seg.pt")

    # Train
    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=4,
        device=0,                 # Use "cpu" if no GPU
        project="runs/train",
        name="BatteryVisionAI",
        pretrained=True,
        optimizer="auto",
        lr0=0.001,
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True
    )

    print("\n===================================")
    print(" Training Completed")
    print("===================================")

    # Validate
    metrics = model.val()

    print("\nValidation Results")
    print(metrics)

    # Export model (optional)
    model.export(format="onnx")

    print("\n===================================")
    print(" Model Exported Successfully")
    print("===================================")


if __name__ == "__main__":
    main()