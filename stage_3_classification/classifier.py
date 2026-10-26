from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class Classifier:
    """ONNX Runtime image-crop classifier."""

    def __init__(
        self,
        model_path: str | Path = "models/classifier.onnx",
        class_names: list[str] | None = None,
        input_size: int = 224,
    ) -> None:
        self.model_path = Path(model_path)
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.class_names = class_names or ["scratch", "dent", "discoloration"]
        self.input_size = input_size
        self.session = None
        self.input_name = ""
        self.input_shape: list[Any] = []
        self.output_name = ""

        if not self.model_path.is_file():
            print(
                f"WARNING: Classifier ONNX model not found at {self.model_path}; "
                "classification will fail if detections require it."
            )
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for classification; install onnxruntime or "
                "onnxruntime-gpu to use the classifier"
            ) from exc

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=ort.get_available_providers(),
        )
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError(f"Classifier model has no inputs: {self.model_path}")
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        outputs = self.session.get_outputs()
        if not outputs:
            raise ValueError(f"Classifier model has no outputs: {self.model_path}")
        self.output_name = outputs[0].name

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            raise ValueError("bbox selects an empty image crop")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )

        if len(self.input_shape) == 4 and self.input_shape[-1] in (3, "3"):
            return normalized[np.newaxis, ...].astype(np.float32)
        return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def _predict_batch(self, crops: list[np.ndarray]) -> list[dict[str, Any]]:
        if not crops:
            return []
        if self.session is None:
            raise FileNotFoundError(
                f"Classifier ONNX model is unavailable: {self.model_path}"
            )
        batch = np.concatenate([self._prepare_crop(crop) for crop in crops], axis=0)
        raw = np.asarray(self.session.run([self.output_name], {self.input_name: batch})[0])
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        if raw.ndim != 2 or raw.shape[0] != len(crops):
            raise ValueError(f"Unexpected classifier output shape: {raw.shape}")

        row_sums = np.sum(raw, axis=1)
        if np.all(raw >= 0) and np.allclose(row_sums, 1.0, atol=1e-4):
            probabilities = raw
        else:
            shifted = raw - np.max(raw, axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        results = []
        for row in probabilities:
            class_index = int(np.argmax(row))
            defect_type = (
                self.class_names[class_index]
                if class_index < len(self.class_names)
                else str(class_index)
            )
            results.append(
                {"defect_type": defect_type, "confidence": float(row[class_index])}
            )
        return results

    @staticmethod
    def _crop(frame: np.ndarray, bbox: list[float] | tuple[float, ...]) -> np.ndarray:
        if len(bbox) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox is outside the frame or has no area: {bbox}")
        return frame[y1:y2, x1:x2]

    def classify(
        self, frame: np.ndarray, bbox: list[float] | tuple[float, ...]
    ) -> dict[str, Any]:
        crop = self._crop(frame, bbox)
        return self._predict_batch([crop])[0]

    def classify_all(
        self, frame: np.ndarray, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        crops = [self._crop(frame, detection["bbox"]) for detection in detections]
        return self._predict_batch(crops)
