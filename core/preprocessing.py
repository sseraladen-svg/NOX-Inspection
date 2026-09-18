"""
core/preprocessing.py

Product-agnostic OpenCV preprocessing stage. All parameters come from the
loaded product config (see config/schema.yaml) — nothing product-specific
is hardcoded here. Swapping products means swapping the config, not this file.

Pipeline: ROI crop (optional) -> resize -> denoise -> CLAHE lighting norm (optional)
"""

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, config: dict):
        """
        config: the full loaded product config dict (see config/schema.yaml).
        Reads config["roi"] and config["preprocessing"].
        """
        self.roi = config.get("roi")

        pre = config["preprocessing"]
        self.resize_dims = tuple(pre["resize"])          # (width, height)
        self.blur_kernel = tuple(pre["blur_kernel"])      # (kx, ky)
        self.use_clahe = pre.get("clahe", False)

        if self.use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def crop_roi(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return frame
        x1, y1, x2, y2 = self.roi
        return frame[y1:y2, x1:x2]

    def resize(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_LINEAR)

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, self.blur_kernel, 0)

    def normalize_lighting(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE on the L channel in LAB space — normalizes lighting without
        distorting color balance. No-op if clahe is disabled in config."""
        if not self.use_clahe:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self._clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline in order: crop -> resize -> denoise -> lighting."""
        frame = self.crop_roi(frame)
        frame = self.resize(frame)
        frame = self.denoise(frame)
        frame = self.normalize_lighting(frame)
        return frame
