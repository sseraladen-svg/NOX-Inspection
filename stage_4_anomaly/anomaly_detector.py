from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class AnomalyDetector:
    """PatchCore-style nearest-memory-bank scoring skeleton.

    The input image is reduced to a grayscale feature vector, so deployments
    should replace `_extract_features` with the same backbone used to build the
    supplied memory bank.
    """

    def __init__(self, memory_bank_path: str | Path = "models/patchcore_model") -> None:
        path = Path(memory_bank_path)
        if path.is_dir():
            candidates = [path / "memory_bank.npy", path / "memory_bank.npz"]
            bank_file = next((candidate for candidate in candidates if candidate.is_file()), None)
            if bank_file is None:
                self.memory_bank = None
                self.memory_bank_path = path
                print(f"WARNING: PatchCore memory bank not found in {path}.")
                return
        else:
            bank_file = path
        if not bank_file.is_file():
            self.memory_bank = None
            self.memory_bank_path = bank_file
            print(f"WARNING: PatchCore memory bank not found: {bank_file}.")
            return

        loaded = np.load(bank_file, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "memory_bank" not in loaded:
                    raise ValueError(f"{bank_file} must contain a 'memory_bank' array")
                memory_bank = loaded["memory_bank"]
            finally:
                loaded.close()
        else:
            memory_bank = loaded
        if memory_bank.size == 0 or memory_bank.ndim < 2:
            raise ValueError("PatchCore memory bank must be a non-empty 2D+ array")
        self.memory_bank = np.asarray(memory_bank, dtype=np.float32).reshape(
            memory_bank.shape[0], -1
        )
        self.memory_bank_path = bank_file

    def _extract_features(self, image_path: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image_path, np.ndarray):
            image = image_path
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.ndim != 2:
                raise ValueError("anomaly image must be grayscale or a 3-channel BGR image")
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read anomaly input image: {image_path}")
        if self.memory_bank is None:
            raise FileNotFoundError(
                f"PatchCore memory bank is unavailable: {self.memory_bank_path}"
            )
        feature_size = self.memory_bank.shape[1]
        side = int(np.sqrt(feature_size))
        target_size = (side, side) if side * side == feature_size else (feature_size, 1)
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        features = resized.astype(np.float32).reshape(1, -1) / 255.0
        return features

    def predict(self, image_path: str | Path | np.ndarray) -> float:
        features = self._extract_features(image_path)
        if self.memory_bank is None:
            raise FileNotFoundError(
                f"PatchCore memory bank is unavailable: {self.memory_bank_path}"
            )
        if features.shape[1] != self.memory_bank.shape[1]:
            raise ValueError("Extracted feature dimension does not match the memory bank")
        distances = np.linalg.norm(self.memory_bank - features, axis=1)
        return float(np.min(distances))

    @staticmethod
    def is_anomalous(score: float, threshold: float) -> bool:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        return score >= threshold
