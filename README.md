# BatteryVisionAI

BatteryVisionAI is a simple computer vision inspection project for detecting defects in battery-related products using deep learning. The project is designed to be easy to understand, test, and extend for students or beginners who want to learn how to build an image-based defect detection pipeline.

## Project Goal

The system can:

- receive images from a camera or upload endpoint
- run defect detection using a pretrained YOLO segmentation model
- save pass/fail results and logs
- provide a simple base for adding a CNN classifier for defect type classification

## What this project does

- accepts image uploads through a Flask server
- runs inference using YOLO
- annotates the image with detection results
- stores results in folders such as:
  - data/pass
  - data/fail
  - data/annotated
  - data/logs

## Main Project Files

- main.py: runs a local image viewer workflow
- server.py: starts the Flask server for image upload and inference
- detection/detector.py: loads the YOLO model and runs detection
- training/train_seg.py: example training script for YOLO segmentation
- cnn_classifier/: folder for building and testing a CNN classifier

## Required Software

This project is built with Python and common machine learning libraries.

Install the main dependencies:

```bash
pip install flask opencv-python numpy ultralytics torch
```

If you are using GPU support, install a PyTorch version that matches your CUDA setup. For most beginners, CPU support is enough to start.

## Model Files

The project currently expects a YOLO segmentation model at:

```text
models/yolo11n-seg.pt
```

If this file is missing, download or place the model in the models folder before running the project.

For a professional setup, the usual workflow is:

- YOLO or similar object detection/segmentation model for locating defects
- CNN classifier for classifying the defect type
- optional anomaly detection model for unseen defects

## Recommended Models

For beginners and practical projects:

- YOLOv8/YOLO11 segmentation: good for detection and segmentation
- ResNet18 or ResNet50: good starting point for CNN classification
- MobileNetV2: lightweight and fast
- EfficientNet: strong balance between speed and accuracy

For a simple first version, start with:

- YOLO for defect detection
- a CNN classifier for defect category recognition

## How a CNN Classifier Can Be Added

A CNN classifier can be added in two main steps:

1. Train the classifier on labeled defect images
2. Integrate the trained model into the inference pipeline

### Suggested folder structure for the CNN classifier

- cnn_classifier/data/: training and validation images
- cnn_classifier/checkpoints/: saved model weights
- cnn_classifier/scripts/: helper scripts for training or evaluation

### Where to integrate it

The classifier can be integrated in the same flow as the main detection pipeline:

- after YOLO detects a defect region, crop the defect area
- pass that crop to the CNN classifier
- return the predicted defect type from the CNN model

In practice, this means the classifier can be called from:

- server.py for uploaded images
- detection/detector.py for inference logic
- main.py for local testing and display

## How to Run the Project

### 1. Start the Flask server

```bash
python server.py
```

Then open:

```text
http://0.0.0.0:5000/upload
```

### 2. Upload an image

Send an image to the upload endpoint. The server will:

- save the original image
- run detection
- create an annotated image
- save results in pass/fail folders

## Training Workflow

If you want to build the CNN classifier:

1. collect labeled images for each defect class
2. place them inside cnn_classifier/data/
3. train the network
4. save the trained weights into cnn_classifier/checkpoints/
5. load the model inside your inference script

A typical training flow is:

- input: labeled images
- output: trained classifier weights
- integration: use weights in the Flask or local pipeline

## Output Folders

The project saves results into the following folders:

- data/original: original uploaded images
- data/annotated: annotated images with detection overlays
- data/pass: accepted images
- data/fail: rejected or defective images
- data/logs: inspection logs and CSV records

## Notes for a Friend Working on the CNN Part

This project is a good starting point because:

- the detection part is already implemented
- the CNN part can be added step by step
- the system is simple enough to understand without too much complexity

A practical approach is:

- first make YOLO detection work correctly
- then add a CNN classifier for defect type classification
- finally connect the classifier to the server output

## License

This project is intended for educational, research, and experimental use.
