import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

class DefectCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DefectCNN, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class CNNPredictor:
    def __init__(self, checkpoint_path=None, class_names=["scratch", "dent", "discoloration"]):
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DefectCNN(num_classes=len(class_names)).to(self.device)
        
        if checkpoint_path and __import__('os').path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
            print(f"[CNN] Loaded weights from {checkpoint_path}")
        else:
            print("[CNN] No weights found. Running in dummy mode.")

    def predict(self, crop_img):
        """Takes a cropped image (numpy array) and returns the predicted class name."""
        if not hasattr(self.model, 'fc') or self.model.fc.weight.shape[0] == 0:
            return "unclassified" # Dummy fallback

        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
        img_norm = (img_resized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        tensor_img = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor_img)
            _, predicted_idx = torch.max(output, 1)
            
        return self.class_names[predicted_idx.item()]