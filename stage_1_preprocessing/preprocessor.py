from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """Prepare inspection frames for downstream inference."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.image_size = int(config.get("image_size", 640))
        self.blur_kernel = int(config.get("blur_kernel", 3))
        self.roi = config.get("roi")
        self.clahe_clip_limit = float(config.get("clahe_clip_limit", 2.0))
        self.clahe_grid_size = int(config.get("clahe_grid_size", 8))
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.blur_kernel < 0 or (self.blur_kernel > 0 and self.blur_kernel % 2 == 0):
            raise ValueError("blur_kernel must be zero or a positive odd number")
        if self.clahe_clip_limit <= 0 or self.clahe_grid_size <= 0:
            raise ValueError("CLAHE clip limit and grid size must be positive")

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return frame
        if isinstance(self.roi, dict):
            x, y = int(self.roi.get("x", 0)), int(self.roi.get("y", 0))
            width = int(self.roi.get("width", frame.shape[1] - x))
            height = int(self.roi.get("height", frame.shape[0] - y))
        elif isinstance(self.roi, (list, tuple)) and len(self.roi) == 4:
            x, y, width, height = (int(value) for value in self.roi)
        else:
            raise ValueError("roi must be {x, y, width, height} or [x, y, width, height]")

        x1, y1 = max(0, x), max(0, y)
        x2 = min(frame.shape[1], x + width)
        y2 = min(frame.shape[0], y + height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"roi does not overlap the input frame: {self.roi}")
        return frame[y1:y2, x1:x2]

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty image array")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a grayscale or 3-channel BGR image")

        processed = self._apply_roi(frame)
        processed = cv2.resize(
            processed,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.blur_kernel > 1:
            processed = cv2.GaussianBlur(processed, (self.blur_kernel, self.blur_kernel), 0)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_grid_size, self.clahe_grid_size),
        )
        enhanced = clahe.apply(lightness)
        processed = cv2.cvtColor(
            cv2.merge((enhanced, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )
        return processed
